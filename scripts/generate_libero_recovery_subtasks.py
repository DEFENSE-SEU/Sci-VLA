#!/usr/bin/env python3
"""Generate recovery-prefixed LIBERO subtask demonstrations.

Pipeline:
1. Read subtask boundaries from split_libero_long_subtasks.py JSON output.
2. For each subtask, keep the original subtask-start scene/object state.
3. Replace only robot joints with a noisy copy of the full-task initial robot state.
4. Servo the robot back to the original subtask-start end-effector pose.
5. Append the original subtask trajectory and write a new hdf5 demo.

The recovery policy emits LIBERO-compatible 7D OSC_POSE actions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


LOW_DIM_OBS_KEYS = ("ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states")
RGB_OBS_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def natural_demo_key(name: str) -> tuple[int, str]:
    match = re.search(r"demo_(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def perturb_robot_qpos(base_qpos: np.ndarray, *, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Return a noisy copy of base_qpos without mutating input."""
    base_qpos = np.asarray(base_qpos, dtype=np.float64)
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")
    return base_qpos + rng.normal(loc=0.0, scale=noise_std, size=base_qpos.shape)


def assemble_recovery_prefixed_segment(
    *,
    recovery_states: np.ndarray,
    recovery_actions: np.ndarray,
    original_states: np.ndarray,
    original_actions: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate recovery and original segment without duplicating target state.

    recovery_states includes both perturbed start and final target-like state.
    recovery_actions has one action per transition, so it must be one shorter
    than recovery_states. The final recovery state is dropped because it is the
    same semantic boundary as original_states[start_frame].
    """
    if len(recovery_actions) != len(recovery_states) - 1:
        raise ValueError("recovery_actions must be one fewer than recovery_states")
    if start_frame < 0 or end_frame >= len(original_states) or start_frame > end_frame:
        raise ValueError("invalid start_frame/end_frame")
    if len(original_states) != len(original_actions):
        raise ValueError("original_states and original_actions must have matching length")

    segment_states = original_states[start_frame : end_frame + 1]
    segment_actions = original_actions[start_frame : end_frame + 1]
    states = np.concatenate([recovery_states[:-1], segment_states], axis=0)
    actions = np.concatenate([recovery_actions, segment_actions], axis=0)
    if len(states) != len(actions):
        raise ValueError("assembled states/actions must have matching length")
    return states, actions


def load_segments(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("segments JSON must contain a list of task records")
    return records


def get_demo_group_names(h5_data_group) -> list[str]:
    return sorted(h5_data_group.keys(), key=natural_demo_key)


def make_env(bddl_file: Path, *, include_rgb: bool, render_gpu_device_id: int):
    # Robosuite's numba cache can fail in some conda / editable installs.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=include_rgb,
        has_renderer=False,
        has_offscreen_renderer=include_rgb,
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=render_gpu_device_id,
    )


def get_robot_refs(env) -> dict[str, np.ndarray]:
    robot = env.env.robots[0]
    refs = {
        "joint_pos": np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64),
        "joint_vel": np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64),
        "joint_model": np.asarray(getattr(robot, "_ref_joint_indexes", []), dtype=np.int64),
        "gripper_pos": np.asarray(getattr(robot, "_ref_gripper_joint_pos_indexes", []) or [], dtype=np.int64),
        "gripper_vel": np.asarray(getattr(robot, "_ref_gripper_joint_vel_indexes", []) or [], dtype=np.int64),
    }
    return refs


def read_robot_qpos_from_state(env, state: np.ndarray, refs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    env.set_init_state(state)
    joint_qpos = env.env.sim.data.qpos[refs["joint_pos"]].copy()
    gripper_qpos = env.env.sim.data.qpos[refs["gripper_pos"]].copy() if len(refs["gripper_pos"]) else np.array([])
    return joint_qpos, gripper_qpos


def clip_joint_qpos(env, qpos: np.ndarray, refs: dict[str, np.ndarray]) -> np.ndarray:
    if len(refs["joint_model"]) != len(qpos):
        return qpos
    ranges = env.env.sim.model.jnt_range[refs["joint_model"]]
    # Some MuJoCo joints may have a zero range for unlimited joints. Panda arm joints are limited.
    lower = ranges[:, 0]
    upper = ranges[:, 1]
    valid = upper > lower
    clipped = qpos.copy()
    clipped[valid] = np.clip(clipped[valid], lower[valid], upper[valid])
    return clipped


def make_perturbed_robot_start_state(
    env,
    *,
    scene_state: np.ndarray,
    base_task_initial_state: np.ndarray,
    refs: dict[str, np.ndarray],
    rng: np.random.Generator,
    robot_noise_std: float,
    keep_subtask_gripper: bool,
) -> np.ndarray:
    base_joint_qpos, base_gripper_qpos = read_robot_qpos_from_state(env, base_task_initial_state, refs)
    target_joint_qpos, target_gripper_qpos = read_robot_qpos_from_state(env, scene_state, refs)

    perturbed_joint_qpos = perturb_robot_qpos(base_joint_qpos, noise_std=robot_noise_std, rng=rng)
    perturbed_joint_qpos = clip_joint_qpos(env, perturbed_joint_qpos, refs)
    gripper_qpos = target_gripper_qpos if keep_subtask_gripper else base_gripper_qpos

    env.set_init_state(scene_state)
    env.env.sim.data.qpos[refs["joint_pos"]] = perturbed_joint_qpos
    env.env.sim.data.qvel[refs["joint_vel"]] = 0.0
    if len(refs["gripper_pos"]) and len(gripper_qpos):
        env.env.sim.data.qpos[refs["gripper_pos"]] = gripper_qpos
        env.env.sim.data.qvel[refs["gripper_vel"]] = 0.0
    env.env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.get_sim_state().copy()


def get_target_ee_pose(env, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = env.set_init_state(state)
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64), np.asarray(obs["robot0_eef_quat"], dtype=np.float64)


def orientation_error_axis_angle(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import robosuite.utils.transform_utils as T
    from robosuite.utils.control_utils import orientation_error

    target_mat = T.quat2mat(target_quat)
    current_mat = T.quat2mat(current_quat)
    return np.asarray(orientation_error(target_mat, current_mat), dtype=np.float64)


def servo_recovery_to_ee_pose(
    env,
    *,
    start_state: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    max_steps: int,
    pos_threshold: float,
    rot_threshold: float,
    pos_gain: float,
    rot_gain: float,
    max_pos_action: float,
    max_rot_action: float,
    gripper_action: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    obs = env.set_init_state(start_state)
    states = [env.get_sim_state().copy()]
    actions = []
    final_pos_error = None
    final_rot_error = None
    success = False

    for step_index in range(max_steps):
        current_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        current_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        pos_error = target_pos - current_pos
        rot_error = orientation_error_axis_angle(target_quat, current_quat)
        final_pos_error = float(np.linalg.norm(pos_error))
        final_rot_error = float(np.linalg.norm(rot_error))
        if final_pos_error <= pos_threshold and final_rot_error <= rot_threshold:
            success = True
            break

        action = np.concatenate(
            [
                np.clip(pos_gain * pos_error, -max_pos_action, max_pos_action),
                np.clip(rot_gain * rot_error, -max_rot_action, max_rot_action),
                np.array([gripper_action], dtype=np.float64),
            ]
        )
        obs, _, _, _ = env.step(action)
        actions.append(action)
        states.append(env.get_sim_state().copy())

    info = {
        "success": success,
        "steps": len(actions),
        "final_pos_error": final_pos_error,
        "final_rot_error": final_rot_error,
    }
    if len(actions) == 0:
        # Preserve the states/actions length contract by emitting one no-op step.
        action = np.zeros(7, dtype=np.float64)
        action[-1] = gripper_action
        obs, _, _, _ = env.step(action)
        actions.append(action)
        states.append(env.get_sim_state().copy())
        info["steps"] = 1
    return np.asarray(states), np.asarray(actions), info


def collect_obs_arrays(env, states: np.ndarray, *, include_rgb: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    obs_values: dict[str, list[np.ndarray]] = {key: [] for key in LOW_DIM_OBS_KEYS}
    if include_rgb:
        for key in RGB_OBS_KEYS:
            obs_values[key] = []
    robot_states = []
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import robosuite.utils.transform_utils as T

    for state in states:
        obs = env.set_init_state(state)
        ee_pos = np.asarray(obs["robot0_eef_pos"])
        ee_ori = np.asarray(T.quat2axisangle(obs["robot0_eef_quat"]))
        obs_values["ee_pos"].append(ee_pos)
        obs_values["ee_ori"].append(ee_ori)
        obs_values["ee_states"].append(np.hstack((ee_pos, ee_ori)))
        obs_values["gripper_states"].append(np.asarray(obs["robot0_gripper_qpos"]))
        obs_values["joint_states"].append(np.asarray(obs["robot0_joint_pos"]))
        if include_rgb:
            obs_values["agentview_rgb"].append(np.asarray(obs["agentview_image"]))
            obs_values["eye_in_hand_rgb"].append(np.asarray(obs["robot0_eye_in_hand_image"]))
        robot_states.append(env.env.get_robot_state_vector(obs))

    return {key: np.asarray(values) for key, values in obs_values.items()}, np.asarray(robot_states)


def write_demo_group(
    data_group,
    *,
    demo_name: str,
    states: np.ndarray,
    actions: np.ndarray,
    obs_arrays: dict[str, np.ndarray],
    robot_states: np.ndarray,
    attrs: dict[str, Any],
) -> None:
    demo_group = data_group.create_group(demo_name)
    demo_group.create_dataset("states", data=states)
    demo_group.create_dataset("actions", data=actions)
    dones = np.zeros(len(states), dtype=np.uint8)
    rewards = np.zeros(len(states), dtype=np.uint8)
    dones[-1] = 1
    rewards[-1] = 1
    demo_group.create_dataset("dones", data=dones)
    demo_group.create_dataset("rewards", data=rewards)
    demo_group.create_dataset("robot_states", data=robot_states)
    obs_group = demo_group.create_group("obs")
    for key, value in obs_arrays.items():
        obs_group.create_dataset(key, data=value)
    for key, value in attrs.items():
        if isinstance(value, (dict, list, tuple)):
            demo_group.attrs[key] = json.dumps(value)
        else:
            demo_group.attrs[key] = value


def source_demo_name_to_segments(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for demo in record["demos"]:
        result[demo["demo_name"]] = demo.get("segments", [])
    return result


def process_task_record(
    record: dict[str, Any],
    *,
    output_root: Path,
    include_rgb: bool,
    seed: int,
    robot_noise_std: float,
    keep_subtask_gripper: bool,
    max_recovery_steps: int,
    pos_threshold: float,
    rot_threshold: float,
    pos_gain: float,
    rot_gain: float,
    max_pos_action: float,
    max_rot_action: float,
    gripper_action: float,
    max_demos: int | None,
    render_gpu_device_id: int,
) -> Path:
    import h5py

    source_demo_file = Path(record["demo_file"])
    bddl_file = Path(record["bddl_file"])
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{record['task_name']}_recovery_subtasks.hdf5"
    rng = np.random.default_rng(seed + int(record["task_id"]) * 100_000)
    segments_by_demo = source_demo_name_to_segments(record)

    env = make_env(bddl_file, include_rgb=include_rgb, render_gpu_device_id=render_gpu_device_id)
    try:
        env.reset()
        refs = get_robot_refs(env)
        with h5py.File(source_demo_file, "r") as source_h5, h5py.File(output_file, "w") as output_h5:
            output_h5.attrs["source_demo_file"] = str(source_demo_file)
            output_h5.attrs["source_segments_record"] = json.dumps(
                {
                    "benchmark_name": record["benchmark_name"],
                    "task_id": record["task_id"],
                    "task_name": record["task_name"],
                    "language": record["language"],
                }
            )
            data_group = output_h5.create_group("data")
            output_demo_index = 0
            for source_demo_index, demo_name in enumerate(get_demo_group_names(source_h5["data"])):
                if max_demos is not None and source_demo_index >= max_demos:
                    break
                segments = segments_by_demo.get(demo_name, [])
                if not segments:
                    continue
                source_demo = source_h5["data"][demo_name]
                original_states = source_demo["states"][()]
                original_actions = source_demo["actions"][()]
                base_task_initial_state = original_states[0]

                for segment in segments:
                    start_frame = int(segment["start_frame"])
                    end_frame = int(segment["end_frame"])
                    target_state = original_states[start_frame]
                    perturbed_start = make_perturbed_robot_start_state(
                        env,
                        scene_state=target_state,
                        base_task_initial_state=base_task_initial_state,
                        refs=refs,
                        rng=rng,
                        robot_noise_std=robot_noise_std,
                        keep_subtask_gripper=keep_subtask_gripper,
                    )
                    target_pos, target_quat = get_target_ee_pose(env, target_state)
                    recovery_states, recovery_actions, recovery_info = servo_recovery_to_ee_pose(
                        env,
                        start_state=perturbed_start,
                        target_pos=target_pos,
                        target_quat=target_quat,
                        max_steps=max_recovery_steps,
                        pos_threshold=pos_threshold,
                        rot_threshold=rot_threshold,
                        pos_gain=pos_gain,
                        rot_gain=rot_gain,
                        max_pos_action=max_pos_action,
                        max_rot_action=max_rot_action,
                        gripper_action=gripper_action,
                    )
                    states, actions = assemble_recovery_prefixed_segment(
                        recovery_states=recovery_states,
                        recovery_actions=recovery_actions,
                        original_states=original_states,
                        original_actions=original_actions,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                    obs_arrays, robot_states = collect_obs_arrays(env, states, include_rgb=include_rgb)
                    write_demo_group(
                        data_group,
                        demo_name=f"demo_{output_demo_index}",
                        states=states,
                        actions=actions,
                        obs_arrays=obs_arrays,
                        robot_states=robot_states,
                        attrs={
                            "source_demo_name": demo_name,
                            "source_start_frame": start_frame,
                            "source_end_frame": end_frame,
                            "subtask_index": int(segment["subtask_index"]),
                            "prompt": segment.get("prompt", ""),
                            "goal_indices": segment.get("goal_indices", []),
                            "recovery_info": recovery_info,
                        },
                    )
                    output_demo_index += 1
            data_group.attrs["num_demos"] = output_demo_index
    finally:
        env.close()
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recovery-prefixed hdf5 demos from LIBERO subtask segment metadata."
    )
    parser.add_argument("--segments", type=Path, required=True, help="JSON from split_libero_long_subtasks.py")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", help="Only process selected task ids.")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-noise-std", type=float, default=0.05)
    parser.add_argument("--max-recovery-steps", type=int, default=50)
    parser.add_argument("--pos-threshold", type=float, default=0.02)
    parser.add_argument("--rot-threshold", type=float, default=0.08)
    parser.add_argument("--pos-gain", type=float, default=8.0)
    parser.add_argument("--rot-gain", type=float, default=2.0)
    parser.add_argument("--max-pos-action", type=float, default=0.6)
    parser.add_argument("--max-rot-action", type=float, default=0.6)
    parser.add_argument("--gripper-action", type=float, default=0.0)
    parser.add_argument("--keep-subtask-gripper", action="store_true")
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_segments(args.segments)
    if args.task_id:
        selected = set(args.task_id)
        records = [record for record in records if int(record["task_id"]) in selected]

    outputs = []
    for record in records:
        outputs.append(
            process_task_record(
                record,
                output_root=args.output_root,
                include_rgb=args.include_rgb,
                seed=args.seed,
                robot_noise_std=args.robot_noise_std,
                keep_subtask_gripper=args.keep_subtask_gripper,
                max_recovery_steps=args.max_recovery_steps,
                pos_threshold=args.pos_threshold,
                rot_threshold=args.rot_threshold,
                pos_gain=args.pos_gain,
                rot_gain=args.rot_gain,
                max_pos_action=args.max_pos_action,
                max_rot_action=args.max_rot_action,
                gripper_action=args.gripper_action,
                max_demos=args.max_demos,
                render_gpu_device_id=args.render_gpu_device_id,
            )
        )

    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
