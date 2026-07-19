from types import SimpleNamespace
from pathlib import Path
import json
import re
import sys

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


def test_rrt_planner_skips_action_when_no_path_is_valid():
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

    assert plan.status == "RRT_FAILED_SKIP_ACTION"
    assert plan.waypoints == []
    assert plan.validation["reason"] == "rrt_failed_skip_action"


def test_rrt_connect_finds_path_around_a_blocked_direct_route():
    from non_llm_transition import plan_joint_path_rrt

    start = np.array([-1.0, 0.0])
    target = np.array([1.0, 0.0])

    def validator(path):
        for first, second in zip(path[:-1], path[1:]):
            for alpha in np.linspace(0.0, 1.0, 101):
                point = first + alpha * (second - first)
                if abs(point[0]) < 0.2 and abs(point[1]) < 0.2:
                    return {"valid": False, "reason": "central_obstacle"}
        return {"valid": True, "reason": "clear"}

    plan = plan_joint_path_rrt(
        start,
        target,
        path_validator=validator,
        joint_ranges=np.array([[-1.5, 1.5], [-1.5, 1.5]]),
        rng=np.random.default_rng(7),
        max_iterations=256,
        step_size=0.2,
        goal_sample_rate=0.2,
        local_sample_rate=0.6,
    )

    assert plan.status == "rrt"
    np.testing.assert_allclose(plan.waypoints[0], start)
    np.testing.assert_allclose(plan.waypoints[-1], target)
    assert len(plan.waypoints) > 2
    assert validator(plan.waypoints)["valid"] is True


def test_rrt_uses_nearest_equivalent_periodic_target_coordinate():
    from non_llm_transition import plan_joint_path_rrt

    start = np.array([3.10])
    target = np.array([-3.10])

    plan = plan_joint_path_rrt(
        start,
        target,
        path_validator=lambda _path: {"valid": True},
        joint_ranges=np.array([[-2.0 * np.pi, 2.0 * np.pi]]),
    )

    assert plan.status == "rrt_direct"
    assert abs(plan.waypoints[-1][0] - start[0]) < 0.2

def test_joint_ranges_from_model_maps_qpos_span_to_scalar_joint_limits():
    from non_llm_transition import joint_ranges_from_model

    model = SimpleNamespace(
        nq=3,
        njnt=3,
        jnt_qposadr=np.array([0, 1, 2], dtype=np.int32),
        jnt_range=np.array(
            [
                [-1.0, 1.0],
                [-2.0, 2.0],
                [-3.0, 3.0],
            ],
            dtype=np.float64,
        ),
    )

    ranges = joint_ranges_from_model(model, range(0, 3))

    np.testing.assert_allclose(
        ranges,
        np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]], dtype=np.float64),
    )


def test_baseline_experiment_mode_uses_random_dataset_task_pose_rrt():
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
    assert config["transition_mode"] == "random_dataset_task_pose_rrt"


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


def test_transition_command_normalization_clamps_motion_steps_to_slow_minimum(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=object),
    )
    from transition_generation import MIN_TRANSITION_MOTION_STEPS, _normalize_transition_commands

    commands = _normalize_transition_commands(
        [
            {"op": "translate", "axis": "z", "distance_m": 0.08, "steps": 12},
            {"op": "rotate", "axis": "x", "angle_deg": 15, "steps": 80},
        ]
    )

    assert commands[0]["steps"] == MIN_TRANSITION_MOTION_STEPS
    assert commands[1]["steps"] == MIN_TRANSITION_MOTION_STEPS


def test_transition_template_commands_use_rrt_motion_without_changing_move_to():
    source = Path("scripts/autobio_scripts/transition_template.py").read_text(encoding="utf-8")

    assert "def move_to_rrt(" in source
    assert "def move_to_target_qpos_rrt(" in source
    assert "joint_ranges_from_model" in source
    assert "joint_ranges=joint_ranges_from_model(self.model, self.jnt_span)" in source
    assert "RRT_FAILED_SKIP_ACTION" in source
    assert "return 0" in source

    translate_body = re.search(
        r"def translate_ee\(.*?\):\n(?P<body>.*?)(?=\n    def )",
        source,
        flags=re.DOTALL,
    ).group("body")
    rotate_body = re.search(
        r"def rotate_ee\(.*?\):\n(?P<body>.*?)(?=\n    def )",
        source,
        flags=re.DOTALL,
    ).group("body")
    move_to_body = re.search(
        r"def move_to\(.*?\):\n(?P<body>.*?)(?=\n    def )",
        source,
        flags=re.DOTALL,
    ).group("body")

    assert "self.move_to_rrt(" in translate_body
    assert "self.move_to_rrt(" in rotate_body
    assert "self.interpolate(" in move_to_body
    assert "self.move_to_rrt(" not in move_to_body


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


def test_random_dataset_task_qpos_samples_without_prompt_matching(tmp_path):
    from non_llm_transition import sample_random_dataset_task_qpos

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
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    selected_qpos, info = sample_random_dataset_task_qpos(
        target_prompt="this prompt should not match anything",
        qpos_db_path=qpos_db_path,
        rng=np.random.default_rng(0),
    )

    assert info["selection_strategy"] == "random_dataset_task_pose"
    assert info["requested_task_prompt"] == "this prompt should not match anything"
    assert info["matched_task_prompt"] is None
    assert info["selected_task_prompt"] in {
        "open the centrifuge lid",
        "close the lid of the thermal cycler",
    }
    assert info["candidate_count"] == 3
    assert selected_qpos.tolist() in [
        [9, 9, 9, 9, 9, 9, 9],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ]
