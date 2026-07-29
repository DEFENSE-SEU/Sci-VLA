import json
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
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


def joint_ranges_from_model(model, jnt_span) -> np.ndarray | None:
    qpos_indices = list(jnt_span)
    if not qpos_indices:
        return None

    qpos_addresses = np.asarray(getattr(model, "jnt_qposadr", []), dtype=np.int64).reshape(-1)
    joint_ranges = np.asarray(getattr(model, "jnt_range", []), dtype=np.float64)
    nq = int(getattr(model, "nq", 0) or 0)
    if qpos_addresses.size == 0 or joint_ranges.ndim != 2 or joint_ranges.shape[1] != 2:
        return None

    ranges_by_qpos: dict[int, np.ndarray] = {}
    sorted_joint_ids = sorted(range(qpos_addresses.size), key=lambda idx: int(qpos_addresses[idx]))
    for order_index, joint_id in enumerate(sorted_joint_ids):
        start = int(qpos_addresses[joint_id])
        if start < 0:
            continue
        if order_index + 1 < len(sorted_joint_ids):
            end = int(qpos_addresses[sorted_joint_ids[order_index + 1]])
        else:
            end = nq
        width = end - start
        if width != 1:
            continue
        low, high = joint_ranges[joint_id]
        if not np.isfinite([low, high]).all() or high <= low:
            continue
        ranges_by_qpos[start] = np.array([low, high], dtype=np.float64)

    ranges = []
    for qpos_index in qpos_indices:
        qpos_index = int(qpos_index)
        if qpos_index not in ranges_by_qpos:
            return None
        ranges.append(ranges_by_qpos[qpos_index])
    return np.asarray(ranges, dtype=np.float64)


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


def _normalize_prompt(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _find_local_prompt_match(
    task_prompt: str,
    prompt_choices: list[str],
    *,
    cutoff: float = 0.72,
) -> tuple[str, float] | None:
    if not prompt_choices:
        return None

    norm_prompt = _normalize_prompt(task_prompt)
    norm_map = {_normalize_prompt(prompt): prompt for prompt in prompt_choices}
    if norm_prompt in norm_map:
        return norm_map[norm_prompt], 1.0

    best_norm = None
    best_score = 0.0
    for choice_norm in norm_map:
        score = SequenceMatcher(None, norm_prompt, choice_norm).ratio()
        if score > best_score:
            best_norm = choice_norm
            best_score = score

    if best_norm is None or best_score < float(cutoff):
        return None
    return norm_map[best_norm], float(best_score)


def _build_task_prompt_index(qpos_db: dict | list) -> dict:
    tasks = qpos_db if isinstance(qpos_db, list) else qpos_db.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Invalid qpos database format: expected a list")

    by_prompt = {}
    for task in tasks:
        prompt = str(task.get("task", task.get("task_prompt", ""))).strip()
        if prompt:
            by_prompt[prompt] = task
    if not by_prompt:
        raise ValueError("No valid task prompts found in qpos database")
    return by_prompt


def _qpos_joint_distance(candidate_qpos, current_joint_pos: np.ndarray) -> float:
    cur = np.asarray(current_joint_pos, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(candidate_qpos, dtype=np.float64).reshape(-1)
    dim = min(6, cur.size, q_arr.size)
    if dim <= 0:
        return float("inf")
    if not np.isfinite(cur[:dim]).all() or not np.isfinite(q_arr[:dim]).all():
        return float("inf")
    return float(np.linalg.norm(cur[:dim] - q_arr[:dim]))


def select_target_qpos_candidate(
    stacked_qpos: list,
    current_joint_pos: np.ndarray,
    *,
    top_k: int = 3,
    path_validator: Callable[..., bool | dict] | None = None,
) -> dict:
    if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
        raise ValueError("No qpos candidates provided")

    effective_top_k = max(1, int(top_k or 3))
    ranked_candidates = []
    for i, qpos in enumerate(stacked_qpos):
        ranked_candidates.append(
            {
                "index": i,
                "distance": _qpos_joint_distance(qpos, current_joint_pos),
                "qpos": np.asarray(qpos, dtype=np.float64).reshape(-1),
            }
        )
    ranked_candidates.sort(key=lambda item: (item["distance"], item["index"]))
    top_candidates = ranked_candidates[: min(effective_top_k, len(ranked_candidates))]

    if path_validator is None:
        selected = top_candidates[0]
        return {
            "selected_index": selected["index"],
            "selected_qpos": selected["qpos"],
            "selected_distance": selected["distance"],
            "top_k": effective_top_k,
            "validation": None,
            "top_candidates": [
                {"index": item["index"], "distance": item["distance"]}
                for item in top_candidates
            ],
        }

    validation_records = [
        {"index": item["index"], "distance": item["distance"]}
        for item in top_candidates
    ]
    for candidate in top_candidates:
        try:
            validation = path_validator(
                candidate["qpos"],
                selected_index=candidate["index"],
            )
        except Exception as e:
            validation = {
                "valid": False,
                "reason": "validator_exception",
                "error": str(e),
            }
        is_valid, validation_payload = _validation_is_valid(validation)
        record_index = next(
            i for i, item in enumerate(validation_records)
            if item["index"] == candidate["index"]
        )
        record = {
            "index": candidate["index"],
            "distance": candidate["distance"],
            "valid": is_valid,
        }
        if validation_payload is not None:
            record.update(validation_payload)
        validation_records[record_index] = record
        if is_valid:
            return {
                "selected_index": candidate["index"],
                "selected_qpos": candidate["qpos"],
                "selected_distance": candidate["distance"],
                "top_k": effective_top_k,
                "validation": record,
                "top_candidates": validation_records,
            }

    fallback = top_candidates[0]
    return {
        "selected_index": fallback["index"],
        "selected_qpos": fallback["qpos"],
        "selected_distance": fallback["distance"],
        "top_k": effective_top_k,
        "validation": {
            "index": fallback["index"],
            "distance": fallback["distance"],
            "valid": False,
            "reason": "validation_failed_fallback",
        },
        "top_candidates": validation_records,
        "fallback_reason": "top_k_validation_exhausted",
    }


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


def _rrt_sampling_bounds(
    start: np.ndarray,
    target: np.ndarray,
    joint_ranges: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if joint_ranges is not None:
        ranges = np.asarray(joint_ranges, dtype=np.float64)
        if ranges.shape[0] >= start.size and ranges.shape[1] == 2:
            lower = ranges[: start.size, 0]
            upper = ranges[: start.size, 1]
            if np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(upper > lower):
                return lower.copy(), upper.copy()

    margin = np.maximum(np.pi, np.abs(target - start) + 0.5)
    lower = np.minimum(start, target) - margin
    upper = np.maximum(start, target) + margin
    return lower, upper


_TWO_PI = 2.0 * np.pi
_PERIODIC_RANGE_MIN_SPAN = _TWO_PI + 1e-3


def _periodic_joint_mask(joint_ranges: np.ndarray | None, dim: int) -> np.ndarray:
    """Identify joints that admit at least one full equivalent revolution."""
    mask = np.zeros(dim, dtype=bool)
    if joint_ranges is None:
        return mask
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if ranges.shape[0] < dim or ranges.shape[1] != 2:
        return mask
    spans = ranges[:dim, 1] - ranges[:dim, 0]
    return np.isfinite(spans) & (spans >= _PERIODIC_RANGE_MIN_SPAN)


def _canonicalize_periodic_qpos(
    qpos: np.ndarray,
    reference: np.ndarray,
    joint_ranges: np.ndarray | None,
) -> np.ndarray:
    """Choose equivalent 2π joint coordinates nearest to a shared reference."""
    canonical = np.asarray(qpos, dtype=np.float64).copy()
    if joint_ranges is None:
        return canonical

    ranges = np.asarray(joint_ranges, dtype=np.float64)
    mask = _periodic_joint_mask(ranges, canonical.size)
    for axis in np.flatnonzero(mask):
        low, high = ranges[axis]
        base = canonical[axis]
        min_turn = int(np.ceil((low - base) / _TWO_PI))
        max_turn = int(np.floor((high - base) / _TWO_PI))
        if min_turn > max_turn:
            continue
        equivalents = base + _TWO_PI * np.arange(min_turn, max_turn + 1)
        canonical[axis] = equivalents[np.argmin(np.abs(equivalents - reference[axis]))]
    return canonical


def _joint_distance(
    first: np.ndarray,
    second: np.ndarray,
    joint_ranges: np.ndarray | None,
) -> float:
    """Euclidean distance in a locally unwrapped joint-coordinate chart."""
    aligned_second = _canonicalize_periodic_qpos(second, first, joint_ranges)
    return float(np.linalg.norm(aligned_second - first))


def _steer_toward(
    start: np.ndarray,
    target: np.ndarray,
    *,
    step_size: float,
    joint_ranges: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    aligned_target = _canonicalize_periodic_qpos(target, start, joint_ranges)
    delta = aligned_target - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return start.copy(), True
    if distance <= step_size:
        return _clamp_to_joint_ranges(aligned_target, joint_ranges), True
    candidate = start + delta / distance * step_size
    return _clamp_to_joint_ranges(candidate, joint_ranges), False


def _sample_rrt_configuration(
    generator: np.random.Generator,
    *,
    start: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_ranges: np.ndarray | None,
    local_sample_rate: float,
) -> np.ndarray:
    """Mix broad exploration with samples in a collision-avoidance corridor."""
    if float(generator.random()) >= local_sample_rate:
        sample = generator.uniform(lower, upper)
    else:
        alpha = float(generator.random())
        center = (1.0 - alpha) * start + alpha * target
        spread = np.maximum(0.35, 0.35 * np.abs(target - start))
        sample = center + generator.normal(scale=spread, size=start.size)
        sample = np.clip(sample, lower, upper)
    return _canonicalize_periodic_qpos(sample, start, joint_ranges)


def _reconstruct_rrt_path(nodes: list[np.ndarray], parents: list[int], end_index: int) -> list[np.ndarray]:
    path = []
    index = end_index
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def plan_joint_path_rrt(
    start_qpos,
    target_qpos,
    *,
    path_validator: PathValidator | None,
    joint_ranges=None,
    rng: np.random.Generator | None = None,
    max_iterations: int = 2048,
    step_size: float = 0.30,
    goal_sample_rate: float = 0.25,
    goal_tolerance: float = 0.12,
    local_sample_rate: float = 0.70,
    connect_max_steps: int = 64,
) -> JointPathPlan:
    """Plan a collision-free joint path with bidirectional RRT-Connect.

    The planner uses a local start-target sampling corridor for efficient
    restoration motions while retaining global samples for obstacle detours.
    Joints with more than one full revolution of valid range are canonicalized
    to the equivalent coordinate nearest to the start state.
    """
    start = _as_joint_vector(start_qpos)
    target = _as_joint_vector(target_qpos, dim=start.size)
    if target.size != start.size:
        raise ValueError(f"Target qpos dim {target.size} does not match start dim {start.size}")

    ranges = None if joint_ranges is None else np.asarray(joint_ranges, dtype=np.float64)
    target = _canonicalize_periodic_qpos(target, start, ranges)
    direct_path = [start, target]
    if path_validator is None:
        return JointPathPlan(waypoints=direct_path, status="rrt_direct_no_validator", validation=None)

    valid, payload = _validation_is_valid(path_validator(direct_path))
    if valid:
        return JointPathPlan(waypoints=direct_path, status="rrt_direct", validation=payload)

    last_payload = payload
    generator = rng if rng is not None else np.random.default_rng()
    lower, upper = _rrt_sampling_bounds(start, target, ranges)
    start_nodes = [start]
    start_parents = [-1]
    goal_nodes = [target]
    goal_parents = [-1]
    iterations = max(1, int(max_iterations))
    edge_step = max(1e-6, float(step_size))
    goal_rate = float(np.clip(goal_sample_rate, 0.0, 1.0))
    local_rate = float(np.clip(local_sample_rate, 0.0, 1.0))
    max_connect = max(1, int(connect_max_steps))

    def extend_tree(nodes, parents, sample):
        distances = np.asarray(
            [_joint_distance(node, sample, ranges) for node in nodes],
            dtype=np.float64,
        )
        nearest_index = int(np.argmin(distances))
        nearest = nodes[nearest_index]
        candidate, reached = _steer_toward(
            nearest,
            sample,
            step_size=edge_step,
            joint_ranges=ranges,
        )
        if np.allclose(candidate, nearest, atol=1e-12, rtol=0.0):
            return None, False, last_payload

        is_valid, edge_payload = _validation_is_valid(path_validator([nearest, candidate]))
        if not is_valid:
            return None, False, edge_payload

        nodes.append(candidate)
        parents.append(nearest_index)
        return len(nodes) - 1, reached, edge_payload

    def connect_tree(nodes, parents, sample):
        edge_payload = last_payload
        for _connect_step in range(max_connect):
            candidate_index, reached, edge_payload = extend_tree(nodes, parents, sample)
            if candidate_index is None:
                return None, edge_payload
            if reached:
                return candidate_index, edge_payload
        return None, edge_payload

    grow_from_start = True
    for iteration in range(iterations):
        if grow_from_start:
            active_nodes, active_parents = start_nodes, start_parents
            other_nodes, other_parents = goal_nodes, goal_parents
            opposite_root = target
        else:
            active_nodes, active_parents = goal_nodes, goal_parents
            other_nodes, other_parents = start_nodes, start_parents
            opposite_root = start

        if float(generator.random()) < goal_rate:
            sample = opposite_root
        else:
            sample = _sample_rrt_configuration(
                generator,
                start=start,
                target=target,
                lower=lower,
                upper=upper,
                joint_ranges=ranges,
                local_sample_rate=local_rate,
            )

        active_index, _reached, payload = extend_tree(active_nodes, active_parents, sample)
        last_payload = payload
        if active_index is not None:
            other_index, payload = connect_tree(
                other_nodes,
                other_parents,
                active_nodes[active_index],
            )
            last_payload = payload
            if other_index is not None:
                if grow_from_start:
                    start_path = _reconstruct_rrt_path(start_nodes, start_parents, active_index)
                    goal_path = _reconstruct_rrt_path(goal_nodes, goal_parents, other_index)
                else:
                    start_path = _reconstruct_rrt_path(start_nodes, start_parents, other_index)
                    goal_path = _reconstruct_rrt_path(goal_nodes, goal_parents, active_index)
                combined_path = start_path + list(reversed(goal_path))[1:]
                path_is_valid, full_path_payload = _validation_is_valid(path_validator(combined_path))
                last_payload = full_path_payload
                if path_is_valid:
                    return JointPathPlan(
                        waypoints=combined_path,
                        status="rrt",
                        validation={
                            "valid": True,
                            "reason": "rrt_connect_path_validated",
                            "iterations": iteration + 1,
                            "start_tree_nodes": len(start_nodes),
                            "goal_tree_nodes": len(goal_nodes),
                            "last_edge_validation": payload,
                            "full_path_validation": full_path_payload,
                        },
                    )

        grow_from_start = not grow_from_start

    failure_payload = {
        "valid": False,
        "reason": "rrt_failed_skip_action",
        "last_validation": last_payload,
        "iterations": iterations,
        "nodes": len(start_nodes) + len(goal_nodes),
        "start_tree_nodes": len(start_nodes),
        "goal_tree_nodes": len(goal_nodes),
        "planner": "bidirectional_rrt_connect",
    }
    return JointPathPlan(
        waypoints=[],
        status="RRT_FAILED_SKIP_ACTION",
        validation=failure_payload,
    )


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
    if not waypoints:
        return 0
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
    if not qpos_db_path.exists():
        raise FileNotFoundError(
            f"Qpos database not found at {qpos_db_path}. "
            "Please run export_lerobot_initial_qpos.py first."
        )
    with open(qpos_db_path, "r", encoding="utf-8") as f:
        qpos_db = json.load(f)

    matched_prompt, stacked_qpos = _stacked_qpos_for_prompt(
        qpos_db,
        target_prompt,
        match_cutoff=match_cutoff,
    )
    selection = select_target_qpos_candidate(
        stacked_qpos,
        current_joint_pos,
        top_k=top_k,
        path_validator=path_validator,
    )
    selected_index = int(selection["selected_index"])
    selected_qpos = stacked_qpos[selected_index]
    selected_qpos_arr = _as_joint_vector(selected_qpos)
    return selected_qpos_arr, {
        "requested_task_prompt": target_prompt,
        "matched_task_prompt": matched_prompt,
        "selected_index": selected_index,
        "candidate_count": len(stacked_qpos),
        "top_k": max(1, int(top_k or 3)),
        "selection": {
            **selection,
            "selected_qpos": _as_joint_vector(selection["selected_qpos"]).tolist(),
        },
    }


def _stacked_qpos_for_prompt(
    qpos_db: dict | list,
    target_prompt: str,
    *,
    match_cutoff: float = 0.5,
) -> tuple[str, list]:
    by_prompt = _build_task_prompt_index(qpos_db)
    local_match = _find_local_prompt_match(
        target_prompt,
        list(by_prompt.keys()),
        cutoff=match_cutoff,
    )
    if local_match is None:
        raise ValueError(f"No task prompt matched for: {target_prompt}")

    matched_prompt, _score = local_match
    matched = by_prompt[matched_prompt]
    stacked_qpos = matched.get("initial_qpos")
    if isinstance(stacked_qpos, list) and len(stacked_qpos) > 0:
        return matched_prompt, stacked_qpos

    entries = matched.get("entries", [])
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"Matched task has no qpos entries: {matched_prompt}")
    stacked_qpos = [
        entry.get("initial_qpos")
        for entry in entries
        if entry.get("initial_qpos") is not None
    ]
    if not stacked_qpos:
        raise ValueError(f"Matched task entries have no initial_qpos: {matched_prompt}")
    return matched_prompt, stacked_qpos


def sample_random_future_task_qpos(
    *,
    target_prompt: str,
    qpos_db_path: Path = Path("logs/lerobot_initial_qpos.json"),
    rng: np.random.Generator | None = None,
    match_cutoff: float = 0.5,
) -> tuple[np.ndarray, dict]:
    if not qpos_db_path.exists():
        raise FileNotFoundError(
            f"Qpos database not found at {qpos_db_path}. "
            "Please run export_lerobot_initial_qpos.py first."
        )
    with open(qpos_db_path, "r", encoding="utf-8") as f:
        qpos_db = json.load(f)

    matched_prompt, stacked_qpos = _stacked_qpos_for_prompt(
        qpos_db,
        target_prompt,
        match_cutoff=match_cutoff,
    )
    generator = rng if rng is not None else np.random.default_rng()
    selected_index = int(generator.integers(0, len(stacked_qpos)))
    selected_qpos = _as_joint_vector(stacked_qpos[selected_index])
    return selected_qpos, {
        "requested_task_prompt": target_prompt,
        "matched_task_prompt": matched_prompt,
        "selected_index": selected_index,
        "candidate_count": len(stacked_qpos),
        "selection_strategy": "random_future_task_pose",
        "selected_qpos": selected_qpos.tolist(),
    }


def _iter_qpos_entries_from_db(qpos_db: dict | list) -> list[dict]:
    tasks = qpos_db if isinstance(qpos_db, list) else qpos_db.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Invalid qpos database format: expected a list")

    entries = []
    for task_index, task_record in enumerate(tasks):
        if not isinstance(task_record, dict):
            continue
        task_prompt = str(task_record.get("task", task_record.get("task_prompt", ""))).strip()
        initial_qpos = task_record.get("initial_qpos")
        if isinstance(initial_qpos, list):
            for qpos_index, qpos in enumerate(initial_qpos):
                if qpos is not None:
                    entries.append(
                        {
                            "task_index": task_index,
                            "task_prompt": task_prompt,
                            "qpos_index": qpos_index,
                            "qpos": qpos,
                        }
                    )

        nested_entries = task_record.get("entries", [])
        if isinstance(nested_entries, list):
            for qpos_index, entry in enumerate(nested_entries):
                if not isinstance(entry, dict) or entry.get("initial_qpos") is None:
                    continue
                entries.append(
                    {
                        "task_index": task_index,
                        "task_prompt": task_prompt,
                        "qpos_index": qpos_index,
                        "qpos": entry["initial_qpos"],
                    }
                )

    if not entries:
        raise ValueError("No initial_qpos entries found in qpos database")
    return entries


def sample_random_dataset_task_qpos(
    *,
    target_prompt: str,
    qpos_db_path: Path = Path("logs/lerobot_initial_qpos.json"),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    if not qpos_db_path.exists():
        raise FileNotFoundError(
            f"Qpos database not found at {qpos_db_path}. "
            "Please run export_lerobot_initial_qpos.py first."
        )
    with open(qpos_db_path, "r", encoding="utf-8") as f:
        qpos_db = json.load(f)

    entries = _iter_qpos_entries_from_db(qpos_db)
    generator = rng if rng is not None else np.random.default_rng()
    selected_entry_index = int(generator.integers(0, len(entries)))
    selected_entry = entries[selected_entry_index]
    selected_qpos = _as_joint_vector(selected_entry["qpos"])
    return selected_qpos, {
        "requested_task_prompt": target_prompt,
        "matched_task_prompt": None,
        "selected_task_prompt": selected_entry["task_prompt"],
        "selected_task_index": int(selected_entry["task_index"]),
        "selected_qpos_index": int(selected_entry["qpos_index"]),
        "selected_index": selected_entry_index,
        "candidate_count": len(entries),
        "selection_strategy": "random_dataset_task_pose",
        "selected_qpos": selected_qpos.tolist(),
    }


def retrieve_ready_memory_initial_qpos(
    *,
    target_prompt: str,
    memory_db_path: Path = Path("logs/ready_memory_index.json"),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Use an exact task match, or randomly fall back to an indexed memory's initial state."""
    if not memory_db_path.exists():
        raise FileNotFoundError(
            f"Ready-memory index not found at {memory_db_path}. "
            "Please run export_lerobot_ready_memory_index.py first."
        )

    from ready_memory_retrieval_agent import (
        _frames_from_memory_entry,
        _load_memory_index,
        _memory_task_prompt,
    )

    memories = _load_memory_index(memory_db_path)
    requested_prompt = str(target_prompt).strip()
    exact_matches = [
        memory for memory in memories if _memory_task_prompt(memory) == requested_prompt
    ]
    if exact_matches:
        memory = exact_matches[0]
        matched_prompt = requested_prompt
        selection_strategy = "exact_ready_memory_initial_pose"
    else:
        candidate_memories = [memory for memory in memories if _memory_task_prompt(memory)]
        if not candidate_memories:
            raise ValueError("Ready-memory index contains no task-labelled memories")
        generator = rng if rng is not None else np.random.default_rng()
        memory = candidate_memories[int(generator.integers(0, len(candidate_memories)))]
        matched_prompt = None
        selection_strategy = "random_ready_memory_initial_pose_fallback"
    frames = _frames_from_memory_entry(memory, base_dir=memory_db_path.parent)
    initial_frame = min(frames, key=lambda frame: int(frame["frame_index"]))
    selected_qpos = _as_joint_vector(initial_frame["state"])
    return selected_qpos, {
        "requested_task_prompt": target_prompt,
        "matched_task_prompt": matched_prompt,
        "selected_task_prompt": _memory_task_prompt(memory),
        "memory_id": memory.get("memory_id", memory.get("id")),
        "episode_index": memory.get("episode_index"),
        "initial_frame_index": int(initial_frame["frame_index"]),
        "selection_strategy": selection_strategy,
        "selected_qpos": selected_qpos.tolist(),
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
    ready_memory_db_path: Path = Path("logs/ready_memory_index.json"),
    rng: np.random.Generator | None = None,
) -> dict:
    if mode not in {
        "retrieval_interp",
        "retrieval_collision_planner",
        "random_future_task_pose_collision_planner",
        "random_future_task_pose_rrt",
        "random_dataset_task_pose_rrt",
        "ready_memory_initial_pose_rrt",
    }:
        raise ValueError(f"Unsupported non-LLM transition mode: {mode}")

    ur_joint_start = model.joint("/ur:shoulder_pan").qposadr.item()
    ur_act_start = model.actuator("/ur:shoulder_pan").id
    gripper_act_id = model.actuator("/ur:2f85:fingers_actuator").id
    jnt_span = range(ur_joint_start, ur_joint_start + 6)
    act_span = range(ur_act_start, ur_act_start + 6)
    current_joint_pos = _as_joint_vector(data.qpos[jnt_span])
    joint_ranges = joint_ranges_from_model(model, jnt_span)
    planning_start = time.perf_counter()

    if mode == "ready_memory_initial_pose_rrt":
        target_qpos, retrieval_info = retrieve_ready_memory_initial_qpos(
            target_prompt=target_prompt,
            memory_db_path=ready_memory_db_path,
            rng=rng,
        )
    elif mode == "random_dataset_task_pose_rrt":
        target_qpos, retrieval_info = sample_random_dataset_task_qpos(
            target_prompt=target_prompt,
            qpos_db_path=qpos_db_path,
            rng=rng,
        )
    elif mode in {"random_future_task_pose_collision_planner", "random_future_task_pose_rrt"}:
        target_qpos, retrieval_info = sample_random_future_task_qpos(
            target_prompt=target_prompt,
            qpos_db_path=qpos_db_path,
            rng=rng,
        )
    else:
        target_qpos, retrieval_info = retrieve_target_qpos_non_llm(
            target_prompt=target_prompt,
            current_joint_pos=current_joint_pos,
            top_k=target_top_k,
            qpos_db_path=qpos_db_path,
            path_validator=None,
        )
    target_arm_qpos = _as_joint_vector(target_qpos, dim=current_joint_pos.size)
    target_gripper = float(target_qpos[-1]) if target_qpos.size > current_joint_pos.size else None

    if mode in {
        "random_future_task_pose_rrt",
        "random_dataset_task_pose_rrt",
        "ready_memory_initial_pose_rrt",
    }:
        path_plan = plan_joint_path_rrt(
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
            rng=rng,
        )
        if path_plan.status == "RRT_FAILED_SKIP_ACTION" or not path_plan.waypoints:
            print(
                "[NonLLMTransition] RRT FAILED; skipping arm transition action "
                f"for target_prompt={target_prompt!r}."
            )
    elif mode in {"retrieval_collision_planner", "random_future_task_pose_collision_planner"}:
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

    if path_plan.status == "RRT_FAILED_SKIP_ACTION" or not path_plan.waypoints:
        executed_steps = 0
    else:
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
