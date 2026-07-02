import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PathValidator = Callable[[list[np.ndarray]], bool | dict]


@dataclass
class JointPathPlan:
    waypoints: list[np.ndarray]
    status: str
    validation: dict | None = None


def _as_joint_vector(qpos, *, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if dim is not None:
        arr = arr[:dim]
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid qpos vector: shape={arr.shape}")
    return arr.copy()


def _validation_is_valid(validation: bool | dict) -> tuple[bool, dict | None]:
    if isinstance(validation, dict):
        for key in ("valid", "success", "is_valid"):
            if key in validation:
                return bool(validation[key]), validation
        return bool(validation), validation
    return bool(validation), None


def _clamp_to_joint_ranges(qpos: np.ndarray, joint_ranges: np.ndarray | None) -> np.ndarray:
    if joint_ranges is None:
        return qpos
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if ranges.shape[0] < qpos.size or ranges.shape[1] != 2:
        return qpos
    return np.clip(qpos, ranges[: qpos.size, 0], ranges[: qpos.size, 1])


def _candidate_waypoints(
    start: np.ndarray,
    target: np.ndarray,
    *,
    joint_ranges: np.ndarray | None,
    max_waypoints: int,
) -> list[np.ndarray]:
    midpoint = 0.5 * (start + target)
    dim = start.size
    offsets = []

    for magnitude in (0.25, -0.25, 0.5, -0.5):
        for axis in range(dim):
            offset = np.zeros(dim, dtype=np.float64)
            offset[axis] = magnitude
            offsets.append(offset)

    paired_offsets = [
        (1, -0.35, 2, 0.35),
        (1, 0.35, 2, -0.35),
        (0, 0.25, 3, -0.25),
        (0, -0.25, 3, 0.25),
        (4, 0.4, 5, -0.4),
        (4, -0.4, 5, 0.4),
    ]
    for a0, v0, a1, v1 in paired_offsets:
        if a0 >= dim or a1 >= dim:
            continue
        offset = np.zeros(dim, dtype=np.float64)
        offset[a0] = v0
        offset[a1] = v1
        offsets.append(offset)

    candidates = []
    seen = set()
    for offset in offsets:
        candidate = _clamp_to_joint_ranges(midpoint + offset, joint_ranges)
        key = tuple(np.round(candidate, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= max(1, int(max_waypoints)):
            break
    return candidates


def plan_joint_path_collision_aware(
    start_qpos,
    target_qpos,
    *,
    path_validator: PathValidator | None,
    joint_ranges=None,
    max_waypoints: int = 32,
) -> JointPathPlan:
    start = _as_joint_vector(start_qpos)
    target = _as_joint_vector(target_qpos, dim=start.size)
    if target.size != start.size:
        raise ValueError(f"Target qpos dim {target.size} does not match start dim {start.size}")

    direct_path = [start, target]
    if path_validator is None:
        return JointPathPlan(waypoints=direct_path, status="direct", validation=None)

    valid, payload = _validation_is_valid(path_validator(direct_path))
    if valid:
        return JointPathPlan(waypoints=direct_path, status="direct", validation=payload)

    last_payload = payload
    for waypoint in _candidate_waypoints(
        start,
        target,
        joint_ranges=None if joint_ranges is None else np.asarray(joint_ranges, dtype=np.float64),
        max_waypoints=max_waypoints,
    ):
        waypoint_path = [start, waypoint, target]
        valid, payload = _validation_is_valid(path_validator(waypoint_path))
        last_payload = payload
        if valid:
            return JointPathPlan(waypoints=waypoint_path, status="waypoint", validation=payload)

    return JointPathPlan(waypoints=direct_path, status="fallback_direct", validation=last_payload)


def interpolate_joint_path(waypoints: list[np.ndarray], *, steps_per_segment: int) -> list[np.ndarray]:
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")
    steps = max(1, int(steps_per_segment))
    path = []
    for i in range(len(waypoints) - 1):
        start = _as_joint_vector(waypoints[i])
        target = _as_joint_vector(waypoints[i + 1], dim=start.size)
        for step in range(1, steps + 1):
            alpha = step / steps
            path.append(start + alpha * (target - start))
    return path


def execute_interpolated_joint_path(
    *,
    task,
    data,
    act_span,
    waypoints: list[np.ndarray],
    steps_per_segment: int = 250,
) -> int:
    controls = interpolate_joint_path(waypoints, steps_per_segment=steps_per_segment)
    action_indices = list(act_span)
    for ctrl in controls:
        data.ctrl[action_indices] = ctrl
        task.step_and_log({})
    return len(controls)


def _contact_pairs(data) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[i]
        pairs.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
    return pairs


def _warning_counts(data) -> np.ndarray:
    warning = getattr(data, "warning", None)
    number = getattr(warning, "number", None)
    if number is None:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(number)


def validate_joint_path_in_mujoco(
    model,
    data,
    jnt_span,
    waypoints: list[np.ndarray],
    *,
    num_steps_per_segment: int = 100,
    allow_existing_contacts: bool = True,
) -> dict:
    import mujoco

    jnt_indices = list(jnt_span)
    if len(jnt_indices) == 0:
        return {"valid": False, "reason": "empty_jnt_span"}

    try:
        sim_data = mujoco.MjData(model)
        sim_data.qpos[:] = data.qpos
        sim_data.qvel[:] = data.qvel
        if getattr(sim_data, "ctrl", None) is not None and getattr(data, "ctrl", None) is not None:
            sim_data.ctrl[:] = data.ctrl

        normalized_waypoints = [
            _as_joint_vector(point, dim=len(jnt_indices))
            for point in waypoints
        ]
        if len(normalized_waypoints) < 2:
            return {"valid": False, "reason": "too_few_waypoints"}

        steps = max(2, int(num_steps_per_segment))
        checked_segments = 0
        for seg_idx in range(len(normalized_waypoints) - 1):
            start = normalized_waypoints[seg_idx]
            target = normalized_waypoints[seg_idx + 1]
            sim_data.qpos[jnt_indices] = start
            sim_data.qvel[:] = 0.0
            mujoco.mj_forward(model, sim_data)
            baseline_contacts = _contact_pairs(sim_data) if allow_existing_contacts else set()

            for alpha in np.linspace(0.0, 1.0, steps):
                sim_data.qpos[jnt_indices] = start + alpha * (target - start)
                sim_data.qvel[:] = 0.0
                mujoco.mj_forward(model, sim_data)

                if not np.isfinite(sim_data.qpos).all() or not np.isfinite(sim_data.qvel).all():
                    return {"valid": False, "reason": "nonfinite_state", "segment": seg_idx}

                warnings = _warning_counts(sim_data)
                if warnings.size > 0 and np.any(warnings):
                    return {
                        "valid": False,
                        "reason": "mujoco_warning",
                        "segment": seg_idx,
                        "warnings": warnings.astype(int).tolist(),
                    }

                new_contacts = _contact_pairs(sim_data) - baseline_contacts
                if new_contacts:
                    return {
                        "valid": False,
                        "reason": "new_collision",
                        "segment": seg_idx,
                        "new_contacts": [list(pair) for pair in sorted(new_contacts)],
                    }
            checked_segments += 1

        return {
            "valid": True,
            "reason": "path_validated",
            "segments": checked_segments,
            "num_steps_per_segment": steps,
        }
    except Exception as e:
        return {"valid": False, "reason": "validator_exception", "error": str(e)}


def _write_transition_log(payload: dict, path: Path = Path("logs/non_llm_transition_selected.json")):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, list):
            serializable[key] = [
                item.tolist() if isinstance(item, np.ndarray) else item
                for item in value
            ]
        else:
            serializable[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[NonLLMTransition] Saved transition selection to: {path}")


def retrieve_target_qpos_non_llm(
    *,
    target_prompt: str,
    current_joint_pos: np.ndarray,
    top_k: int = 3,
    qpos_db_path: Path = Path("logs/lerobot_initial_qpos.json"),
    path_validator=None,
    match_cutoff: float = 0.72,
) -> tuple[np.ndarray, dict]:
    from transition_generation import _fallback_find_qpos

    if not qpos_db_path.exists():
        raise FileNotFoundError(
            f"Qpos database not found at {qpos_db_path}. "
            "Please run export_lerobot_initial_qpos.py first."
        )
    with open(qpos_db_path, "r", encoding="utf-8") as f:
        qpos_db = json.load(f)

    matched_prompt, selected_qpos, candidate_count, selected_index, selection = _fallback_find_qpos(
        qpos_db,
        target_prompt,
        current_joint_pos,
        top_k=top_k,
        path_validator=path_validator,
        match_cutoff=match_cutoff,
    )
    selected_qpos_arr = _as_joint_vector(selected_qpos)
    return selected_qpos_arr, {
        "requested_task_prompt": target_prompt,
        "matched_task_prompt": matched_prompt,
        "selected_index": selected_index,
        "candidate_count": candidate_count,
        "top_k": max(1, int(top_k or 3)),
        "selection": {
            **selection,
            "selected_qpos": _as_joint_vector(selection["selected_qpos"]).tolist(),
        },
    }


def execute_non_llm_transition(
    *,
    model,
    data,
    task,
    target_prompt: str,
    mode: str = "retrieval_collision_planner",
    target_top_k: int = 3,
    restore_steps_per_segment: int = 250,
    validation_steps_per_segment: int = 100,
    qpos_db_path: Path = Path("logs/lerobot_initial_qpos.json"),
) -> dict:
    if mode not in {"retrieval_interp", "retrieval_collision_planner"}:
        raise ValueError(f"Unsupported non-LLM transition mode: {mode}")

    ur_joint_start = model.joint("/ur:shoulder_pan").qposadr.item()
    ur_act_start = model.actuator("/ur:shoulder_pan").id
    gripper_act_id = model.actuator("/ur:2f85:fingers_actuator").id
    jnt_span = range(ur_joint_start, ur_joint_start + 6)
    act_span = range(ur_act_start, ur_act_start + 6)
    current_joint_pos = _as_joint_vector(data.qpos[jnt_span])
    joint_ranges = None
    planning_start = time.perf_counter()

    target_qpos, retrieval_info = retrieve_target_qpos_non_llm(
        target_prompt=target_prompt,
        current_joint_pos=current_joint_pos,
        top_k=target_top_k,
        qpos_db_path=qpos_db_path,
        path_validator=None,
    )
    target_arm_qpos = _as_joint_vector(target_qpos, dim=current_joint_pos.size)
    target_gripper = float(target_qpos[-1]) if target_qpos.size > current_joint_pos.size else None

    if mode == "retrieval_collision_planner":
        path_plan = plan_joint_path_collision_aware(
            current_joint_pos,
            target_arm_qpos,
            path_validator=lambda path: validate_joint_path_in_mujoco(
                model,
                data,
                jnt_span,
                path,
                num_steps_per_segment=validation_steps_per_segment,
            ),
            joint_ranges=joint_ranges,
        )
    else:
        path_plan = JointPathPlan(
            waypoints=[current_joint_pos, target_arm_qpos],
            status="direct_no_collision_planner",
            validation=None,
        )
    planning_elapsed = time.perf_counter() - planning_start

    executed_steps = execute_interpolated_joint_path(
        task=task,
        data=data,
        act_span=act_span,
        waypoints=path_plan.waypoints,
        steps_per_segment=restore_steps_per_segment,
    )

    if target_gripper is not None:
        data.ctrl[gripper_act_id] = target_gripper
        for _ in range(50):
            task.step_and_log({})

    result = {
        "mode": mode,
        "target_prompt": target_prompt,
        "retrieval": retrieval_info,
        "planner_status": path_plan.status,
        "planner_validation": path_plan.validation,
        "waypoints": path_plan.waypoints,
        "target_arm_qpos": target_arm_qpos,
        "target_gripper": target_gripper,
        "executed_steps": executed_steps,
        "planning_elapsed": planning_elapsed,
    }
    _write_transition_log(result)
    print(
        "[NonLLMTransition] "
        f"mode={mode} matched={retrieval_info['matched_task_prompt']!r} "
        f"planner={path_plan.status} steps={executed_steps}"
    )
    return result
