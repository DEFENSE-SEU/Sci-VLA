from types import SimpleNamespace

import numpy as np


def test_collision_aware_planner_uses_direct_path_when_valid():
    from non_llm_transition import plan_joint_path_collision_aware

    start = np.zeros(6)
    target = np.ones(6)
    seen_paths = []

    def validator(path):
        seen_paths.append([point.copy() for point in path])
        return {"valid": True, "reason": "path_validated"}

    plan = plan_joint_path_collision_aware(start, target, path_validator=validator)

    assert plan.status == "direct"
    assert len(plan.waypoints) == 2
    np.testing.assert_allclose(plan.waypoints[0], start)
    np.testing.assert_allclose(plan.waypoints[-1], target)
    assert len(seen_paths) == 1


def test_collision_aware_planner_falls_back_to_deterministic_waypoint():
    from non_llm_transition import plan_joint_path_collision_aware

    start = np.zeros(6)
    target = np.ones(6)

    def validator(path):
        return {"valid": len(path) == 3, "reason": "fake_validator"}

    plan = plan_joint_path_collision_aware(start, target, path_validator=validator)

    assert plan.status == "waypoint"
    assert len(plan.waypoints) == 3
    np.testing.assert_allclose(plan.waypoints[0], start)
    np.testing.assert_allclose(plan.waypoints[-1], target)


def test_execute_interpolated_joint_path_drives_target_controls():
    from non_llm_transition import execute_interpolated_joint_path

    data = SimpleNamespace(
        ctrl=np.zeros(7, dtype=np.float64),
        qpos=np.zeros(7, dtype=np.float64),
    )
    task = SimpleNamespace(step_count=0)

    def step_and_log(_info):
        task.step_count += 1

    task.step_and_log = step_and_log
    waypoints = [np.zeros(6), np.full(6, 0.5), np.ones(6)]

    steps = execute_interpolated_joint_path(
        task=task,
        data=data,
        act_span=range(6),
        waypoints=waypoints,
        steps_per_segment=4,
    )

    assert steps == 8
    assert task.step_count == 8
    np.testing.assert_allclose(data.ctrl[:6], np.ones(6))
