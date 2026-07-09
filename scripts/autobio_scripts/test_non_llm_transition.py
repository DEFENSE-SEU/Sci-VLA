from types import SimpleNamespace
from pathlib import Path
import json

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


def test_rrt_planner_falls_back_to_interpolation_when_no_path_is_valid():
    from non_llm_transition import plan_joint_path_rrt

    start = np.zeros(6)
    target = np.ones(6)

    def validator(_path):
        return {"valid": False, "reason": "blocked"}

    plan = plan_joint_path_rrt(
        start,
        target,
        path_validator=validator,
        rng=np.random.default_rng(0),
        max_iterations=4,
    )

    assert "FALLBACK" in plan.status
    assert len(plan.waypoints) == 2
    np.testing.assert_allclose(plan.waypoints[0], start)
    np.testing.assert_allclose(plan.waypoints[-1], target)


def test_baseline_experiment_mode_uses_random_future_task_pose_rrt():
    from evaluate import resolve_experiment_mode_config

    config = resolve_experiment_mode_config(
        experiment_mode="baseline",
        use_transition_generation=False,
        transition_mode="auto",
        no_planning=False,
        no_interpolation=False,
        no_retrieval=False,
    )

    assert config["use_transition_generation"] is True
    assert config["transition_mode"] == "random_future_task_pose_rrt"


def test_evaluate_wires_no_render_video_flag_to_evaluator():
    source = Path("scripts/autobio_scripts/evaluate.py").read_text(encoding="utf-8")

    assert '"--render-video"' in source
    assert "argparse.BooleanOptionalAction" in source
    assert "render_video=args.render_video" in source
    assert "render_video: bool" in source
    assert "args.render_video," in source


def test_evaluator_can_skip_replay_video_capture_and_save():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    assert "render_video: bool = True" in source
    assert "self.render_video = bool(render_video)" in source
    assert "if not self.render_video:" in source
    assert "Replay video rendering disabled; skipping video save." in source


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


def test_random_future_task_qpos_samples_only_from_matched_task(tmp_path):
    from non_llm_transition import sample_random_future_task_qpos

    qpos_db_path = tmp_path / "lerobot_initial_qpos.json"
    qpos_db_path.write_text(
        json.dumps(
            [
                {
                    "task": "open the centrifuge lid",
                    "initial_qpos": [[9, 9, 9, 9, 9, 9, 9]],
                },
                {
                    "task": "close the lid of the thermal cycler",
                    "initial_qpos": [
                        [0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2, 2, 2],
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    selected_qpos, info = sample_random_future_task_qpos(
        target_prompt="close the lid of the thermal cycler",
        qpos_db_path=qpos_db_path,
        rng=np.random.default_rng(0),
    )

    assert info["selection_strategy"] == "random_future_task_pose"
    assert info["matched_task_prompt"] == "close the lid of the thermal cycler"
    assert info["candidate_count"] == 3
    assert info["selected_index"] in {0, 1, 2}
    assert selected_qpos.tolist() in [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2],
    ]
