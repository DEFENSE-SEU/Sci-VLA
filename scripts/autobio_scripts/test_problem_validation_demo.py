import json
import math
import re
import sys
from argparse import Namespace
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluator as evaluator_module
from evaluator import Evaluator, restore_robot_state_direct
from problem_validation_demo import (
    ProblemValidationDemoConfig,
    PromptRunController,
    SampledRobotState,
    execute_problem_validation_sequence,
    sample_problem_validation_state,
)
from evaluate import apply_problem_validation_demo_profile, evaluate_task, parse_args


class _FakeTask:
    def __init__(self):
        self.steps = 0

    def step_and_log(self, _info):
        self.steps += 1


class _FakeData:
    def __init__(self):
        self.qpos = np.zeros(7, dtype=np.float64)
        self.ctrl = np.zeros(7, dtype=np.float64)


def test_restore_robot_state_interpolates_arm_and_gripper_together():
    task = _FakeTask()
    data = _FakeData()
    target = np.arange(1.0, 8.0)

    steps = restore_robot_state_direct(
        task=task,
        data=data,
        state_indices=range(7),
        action_indices=range(7),
        target_state=target,
        num_steps=4,
    )

    assert steps == 4
    assert task.steps == 4
    np.testing.assert_allclose(data.ctrl, target)


def test_restore_robot_state_does_not_modify_non_robot_qpos():
    task = _FakeTask()
    data = _FakeData()
    data.qpos = np.arange(10.0, 19.0)
    original_qpos = data.qpos.copy()

    restore_robot_state_direct(
        task=task,
        data=data,
        state_indices=range(1, 8),
        action_indices=range(7),
        target_state=np.arange(1.0, 8.0),
        num_steps=2,
    )

    np.testing.assert_array_equal(data.qpos, original_qpos)


def _make_fake_evaluator(check_prompt):
    events = {
        "captures": 0,
        "finishes": 0,
        "render_finishes": 0,
        "videos": [],
        "policy_prompts": [],
        "policy_ctrl": [],
        "prompt_starts": [],
    }
    data = SimpleNamespace(
        time=0.0,
        qpos=np.zeros(7, dtype=np.float64),
        ctrl=np.zeros(7, dtype=np.float64),
        warning=SimpleNamespace(number=np.zeros(1, dtype=np.int64)),
    )

    class FakeTask:
        time_limit = 30.0
        task_info = {
            "state_indices": range(7),
            "action_indices": range(7),
            "prefix": None,
        }

        def step_and_log(self, _info):
            data.time += 1.0

        def check(self, prompt=None):
            return check_prompt(prompt, data.time)

        def record_atomic_start(self, prompt):
            events["prompt_starts"].append(prompt)

        def finish(self):
            events["finishes"] += 1

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.task = FakeTask()
    evaluator.model = SimpleNamespace(
        opt=SimpleNamespace(timestep=1.0),
        joint=lambda _name: SimpleNamespace(qposadr=np.asarray(0)),
    )
    evaluator.data = data
    evaluator.task_info = evaluator.task.task_info.copy()
    evaluator.history_states = deque()
    evaluator.reset = lambda: None
    evaluator.get_observation = lambda: {}

    def capture():
        events["captures"] += 1

    def render_finish():
        events["render_finishes"] += 1

    def save_video(success, filename_override=None, action_count=None):
        events["videos"].append((success, filename_override, action_count))

    evaluator._capture_replay_frame = capture
    evaluator.render_finish = render_finish
    evaluator.save_video = save_video

    def policy(_observation):
        events["policy_prompts"].append(evaluator.task_info["prefix"])
        events["policy_ctrl"].append(data.ctrl.copy())
        return np.zeros((1, 7), dtype=np.float64)

    return evaluator, policy, events, data


def test_ordinary_prompt_still_stops_on_first_success():
    evaluator, policy, events, data = _make_fake_evaluator(
        lambda _prompt, current_time: current_time >= 1.0
    )

    success, timing = evaluator.evaluate(
        policy,
        time_limit=10.0,
        prompts=["ordinary prompt"],
        control_fps=1.0,
    )

    assert success
    assert data.time == 2.0  # one existing pre-action settle step plus one policy action
    assert events["policy_prompts"] == ["ordinary prompt"]
    assert timing["atomic_task_results"][0]["success"] is True


def test_ordinary_timeout_still_uses_final_predicate_state():
    evaluator, policy, events, _data = _make_fake_evaluator(
        lambda _prompt, current_time: current_time == 2.0
    )

    success, timing = evaluator.evaluate(
        policy,
        time_limit=3.0,
        prompts=["ordinary timeout prompt"],
        control_fps=1.0,
        intervention_mode="timeout",
    )

    assert not success
    assert events["policy_prompts"] == ["ordinary timeout prompt"] * 3
    assert timing["atomic_task_results"] == [
        {
            "prompt_index": 0,
            "prompt": "ordinary timeout prompt",
            "success": False,
            "attempt_index": 0,
        }
    ]


def test_ordinary_multi_prompt_path_still_runs_each_prompt_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    evaluator, policy, events, _data = _make_fake_evaluator(
        lambda _prompt, current_time: current_time >= 1.0
    )
    evaluator.get_transition_views = lambda: (None, None)

    success, timing = evaluator.evaluate(
        policy,
        time_limit=10.0,
        prompts=["first ordinary prompt", "second ordinary prompt"],
        control_fps=1.0,
        use_transition_generation=False,
        transition_mode="none",
    )

    assert success
    assert events["policy_prompts"] == ["first ordinary prompt", "second ordinary prompt"]
    assert [result["success"] for result in timing["atomic_task_results"]] == [True, True]
    assert events["finishes"] == events["render_finishes"] == 1


def test_demo_forces_non_timeout_tail_and_stops_inside_action_chunk(
    monkeypatch,
    capsys,
    tmp_path,
):
    config = replace(
        ProblemValidationDemoConfig(dataset_root=tmp_path),
        post_success_seconds=2.5,
        interpolation_steps=1,
    )
    evaluator, _default_policy, events, data = _make_fake_evaluator(
        lambda prompt, current_time: (
            current_time == 2.0
            if prompt == config.prompts[0]
            else current_time == 7.0
        )
    )
    sampled = SampledRobotState(
        state=np.arange(1.0, 8.0),
        episode_index=7,
        episode_length=10,
        prefix_frame_count=3,
        frame_index=2,
        frame_ratio=0.2,
        task=config.prompts[1],
        dataset_root=tmp_path,
    )
    monkeypatch.setattr(evaluator_module, "sample_problem_validation_state", lambda *_: sampled)
    monkeypatch.setattr(
        evaluator_module,
        "count_robot_object_collision_contacts",
        lambda _model, _data: 0,
    )

    def chunked_policy(_observation):
        events["policy_prompts"].append(evaluator.task_info["prefix"])
        return np.zeros((8, 7), dtype=np.float64)

    success, timing = evaluator.evaluate(
        chunked_policy,
        time_limit=8.0,
        prompts=list(config.prompts),
        control_fps=1.0,
        intervention_mode="timeout",
        problem_validation_demo_config=config,
    )

    output = capsys.readouterr().out
    assert success
    assert events["policy_prompts"] == list(config.prompts)
    assert [result["success"] for result in timing["atomic_task_results"]] == [True, True]
    assert events["videos"] == [(True, config.video_filename_prefix, 5)]
    assert data.time == 7.0
    tail_elapsed = float(re.search(r"tail_complete_time=.* elapsed=([0-9.]+)", output).group(1))
    assert tail_elapsed >= config.post_success_seconds
    assert tail_elapsed - config.post_success_seconds <= 1.0


def test_demo_runs_latched_tail_restore_and_second_prompt_once(monkeypatch, capsys, tmp_path):
    config = replace(
        ProblemValidationDemoConfig(dataset_root=tmp_path),
        interpolation_steps=3,
    )
    evaluator, policy, events, data = _make_fake_evaluator(
        lambda prompt, current_time: (
            current_time >= 1.0 if prompt == config.prompts[0] else prompt == config.prompts[1]
        )
    )
    rng_draws = []

    def fake_sample(sample_config, rng):
        assert sample_config is config
        rng_draws.append(int(rng.integers(1_000_000)))
        return SampledRobotState(
            state=np.arange(1.0, 8.0),
            episode_index=7,
            episode_length=10,
            prefix_frame_count=3,
            frame_index=2,
            frame_ratio=0.2,
            task=config.prompts[1],
            dataset_root=tmp_path,
        )

    monkeypatch.setattr(evaluator_module, "sample_problem_validation_state", fake_sample)
    monkeypatch.setattr(
        evaluator_module,
        "count_robot_object_collision_contacts",
        lambda _model, _data: 1,
    )

    success, timing = evaluator.evaluate(
        policy,
        time_limit=10.0,
        prompts=list(config.prompts),
        control_fps=1.0,
        transition_seed=23,
        problem_validation_demo_config=config,
    )

    output = capsys.readouterr().out
    assert success
    assert events["policy_prompts"] == [config.prompts[0]] * 6 + [config.prompts[1]]
    assert rng_draws == [int(np.random.default_rng(23).integers(1_000_000))]
    assert events["captures"] == 11
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(True, config.video_filename_prefix, 7)]
    assert [result["prompt"] for result in timing["atomic_task_results"]] == list(config.prompts)
    assert timing["transition_count"] == 1
    assert timing["transition_collision_counts"] == {"1": 3}
    assert output.count("[ProblemValidationDemo] first_success_time=") == 1
    assert output.count("[ProblemValidationDemo] tail_complete_time=") == 1
    assert "episode=7" in output
    assert "length=10" in output
    assert "prefix_size=3" in output
    assert "frame=2" in output
    assert "ratio=0.200000" in output
    assert "state=[1. 2. 3. 4. 5. 6. 7.]" in output
    assert "restore_steps=3" in output
    np.testing.assert_allclose(events["policy_ctrl"][-1], np.arange(1.0, 8.0))


def test_demo_first_prompt_failure_does_not_sample(monkeypatch, tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, policy, events, _data = _make_fake_evaluator(lambda _prompt, _time: False)
    monkeypatch.setattr(
        evaluator_module,
        "sample_problem_validation_state",
        lambda *_: pytest.fail("must not sample after first prompt failure"),
    )

    success, timing = evaluator.evaluate(
        policy,
        time_limit=2.0,
        prompts=list(config.prompts),
        control_fps=1.0,
        problem_validation_demo_config=config,
    )

    assert not success
    assert len(timing["atomic_task_results"]) == 1
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 2)]


def test_demo_sampling_error_saves_failure_video_once_then_reraises(monkeypatch, tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, policy, events, _data = _make_fake_evaluator(
        lambda prompt, current_time: prompt == config.prompts[0] and current_time >= 1.0
    )
    error = RuntimeError("sample exploded")

    def fail_sample(*_args):
        raise error

    monkeypatch.setattr(evaluator_module, "sample_problem_validation_state", fail_sample)

    with pytest.raises(RuntimeError) as exc_info:
        evaluator.evaluate(
            policy,
            time_limit=10.0,
            prompts=list(config.prompts),
            control_fps=1.0,
            problem_validation_demo_config=config,
        )

    assert exc_info.value is error
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 6)]


def test_demo_restore_error_saves_failure_video_once_then_reraises(monkeypatch, tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, policy, events, _data = _make_fake_evaluator(
        lambda prompt, current_time: prompt == config.prompts[0] and current_time >= 1.0
    )
    sampled = SampledRobotState(
        state=np.arange(1.0, 8.0),
        episode_index=7,
        episode_length=10,
        prefix_frame_count=3,
        frame_index=2,
        frame_ratio=0.2,
        task=config.prompts[1],
        dataset_root=tmp_path,
    )
    error = RuntimeError("restore exploded")
    monkeypatch.setattr(evaluator_module, "sample_problem_validation_state", lambda *_: sampled)

    def fail_restore(**_kwargs):
        raise error

    monkeypatch.setattr(evaluator_module, "restore_robot_state_direct", fail_restore)

    with pytest.raises(RuntimeError) as exc_info:
        evaluator.evaluate(
            policy,
            time_limit=10.0,
            prompts=list(config.prompts),
            control_fps=1.0,
            problem_validation_demo_config=config,
        )

    assert exc_info.value is error
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 6)]


def test_demo_settle_failure_uses_demo_video_name_once(tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, policy, events, data = _make_fake_evaluator(lambda _prompt, _time: False)
    data.warning.number[:] = 1

    success, timing = evaluator.evaluate(
        policy,
        time_limit=10.0,
        prompts=list(config.prompts),
        control_fps=1.0,
        problem_validation_demo_config=config,
    )

    assert not success
    assert timing["atomic_task_results"] == []
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 0)]


def test_demo_settle_failure_attempts_all_cleanup_when_finish_raises(tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, policy, events, data = _make_fake_evaluator(lambda _prompt, _time: False)
    data.warning.number[:] = 1
    finish_error = RuntimeError("finish exploded")

    def fail_finish():
        events["finishes"] += 1
        raise finish_error

    evaluator.task.finish = fail_finish

    success, timing = evaluator.evaluate(
        policy,
        time_limit=10.0,
        prompts=list(config.prompts),
        control_fps=1.0,
        problem_validation_demo_config=config,
    )

    assert not success
    assert timing["atomic_task_results"] == []
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 0)]


def test_demo_policy_error_records_failed_attempt_and_preserves_error(capsys, tmp_path):
    config = replace(ProblemValidationDemoConfig(dataset_root=tmp_path), interpolation_steps=2)
    evaluator, _policy, events, _data = _make_fake_evaluator(lambda _prompt, _time: False)
    policy_error = RuntimeError("policy exploded")
    cleanup_error = RuntimeError("finish exploded")

    def fail_finish():
        events["finishes"] += 1
        raise cleanup_error

    evaluator.task.finish = fail_finish

    def fail_policy(_observation):
        raise policy_error

    with pytest.raises(RuntimeError) as exc_info:
        evaluator.evaluate(
            fail_policy,
            time_limit=10.0,
            prompts=list(config.prompts),
            control_fps=1.0,
            problem_validation_demo_config=config,
        )

    output = capsys.readouterr().out
    assert exc_info.value is policy_error
    assert events["prompt_starts"] == [config.prompts[0]]
    assert events["finishes"] == events["render_finishes"] == 1
    assert events["videos"] == [(False, config.video_filename_prefix, 0)]
    assert (
        "prompt_index=0 attempt_index=0 healthy=False success=False "
        f"prompt={config.prompts[0]!r}"
    ) in output


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_demo_profile_overrides_conflicting_generic_evaluate_args():
    args = Namespace(
        problem_validation_demo=True,
        task="pickup",
        prompts="wrong,prompts",
        num_episodes=9,
        num_workers=4,
        render_video=False,
        intervention_mode="timeout",
        experiment_mode="full",
        use_transition_generation=True,
        transition_mode="llm",
        no_planning=True,
        no_interpolation=True,
        no_retrieval=True,
    )

    config = apply_problem_validation_demo_profile(args)

    assert config is not None
    assert args.task == config.task_name
    assert args.prompts == ",".join(config.prompts)
    assert args.num_episodes == 1
    assert args.num_workers == 0
    assert args.render_video is True
    assert args.intervention_mode == "non_timeout"
    assert args.experiment_mode == "no-transition"
    assert args.use_transition_generation is False
    assert args.transition_mode == "none"
    assert not args.no_planning and not args.no_interpolation and not args.no_retrieval


def test_parse_args_accepts_problem_validation_demo_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--problem-validation-demo"])

    args = parse_args()

    assert args.problem_validation_demo is True


def test_readme_documents_problem_validation_demo_contract():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")

    for expected_text in (
        "--problem-validation-demo",
        "open the lid of the thermal cycler",
        "place pcrPlate into the thermal cycler",
        "mani_thermalcycler",
        "problem_validation_open_lid_place_pcr_plate",
    ):
        assert expected_text in readme


def test_evaluate_task_forwards_problem_validation_demo_config():
    class FakeTask:
        def reset(self, seed):
            self.seed = seed

    class FakeEvaluator:
        def __init__(self):
            self.task = FakeTask()
            self.kwargs = None

        def evaluate(self, *args, **kwargs):
            self.kwargs = kwargs
            return True

    evaluator = FakeEvaluator()
    config = ProblemValidationDemoConfig()

    evaluate_task(
        evaluator,
        object(),
        seed=17,
        time_limit=30.0,
        problem_validation_demo_config=config,
    )

    assert evaluator.task.seed == 17
    assert evaluator.kwargs["problem_validation_demo_config"] is config


def _write_episode(
    root: Path,
    episode_index: int,
    states: list[list[float]],
    *,
    frame_indices=None,
):
    path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    state_type = pa.list_(pa.float32(), len(states[0]))
    frame_indices = range(len(states)) if frame_indices is None else frame_indices
    pq.write_table(
        pa.table(
            {
                "state": pa.array(states, type=state_type),
                "frame_index": pa.array(frame_indices),
            }
        ),
        path,
    )


def _demo_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "mani_thermalcycler"
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 4, "tasks": ["open the lid of the thermal cycler"], "length": 10},
            {"episode_index": 7, "tasks": ["place pcrPlate into the thermal cycler"], "length": 10},
        ],
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    _write_episode(root, 4, [[-1.0] * 7 for _ in range(10)])
    _write_episode(root, 7, [[float(frame)] * 7 for frame in range(10)])
    return root


def test_samples_reproducible_state_from_first_thirty_percent(tmp_path):
    root = _demo_dataset(tmp_path)
    config = ProblemValidationDemoConfig(dataset_root=root)

    first = sample_problem_validation_state(config, np.random.default_rng(19))
    second = sample_problem_validation_state(config, np.random.default_rng(19))

    assert first == second
    assert first.episode_index == 7
    assert first.prefix_frame_count == math.ceil(0.30 * 10)
    assert 0 <= first.frame_index < first.prefix_frame_count
    np.testing.assert_allclose(first.state, [float(first.frame_index)] * 7)


def test_sampler_rejects_nonfinite_state(tmp_path):
    root = _demo_dataset(tmp_path)
    _write_episode(root, 7, [[0.0] * 6 + [float("nan")]] * 10)

    with pytest.raises(ValueError, match="finite 7-dimensional"):
        sample_problem_validation_state(
            ProblemValidationDemoConfig(dataset_root=root),
            np.random.default_rng(0),
        )


def test_sampler_rejects_six_dimensional_state(tmp_path):
    root = _demo_dataset(tmp_path)
    _write_episode(root, 7, [[0.0] * 6 for _ in range(10)])

    with pytest.raises(ValueError, match="finite 7-dimensional"):
        sample_problem_validation_state(
            ProblemValidationDemoConfig(dataset_root=root),
            np.random.default_rng(0),
        )


def test_sampler_rejects_duplicate_prefix_frame_index(tmp_path):
    root = _demo_dataset(tmp_path)
    _write_episode(
        root,
        7,
        [[float(frame)] * 7 for frame in range(10)],
        frame_indices=[0, 0, *range(2, 10)],
    )

    with pytest.raises(ValueError, match=r"Episode 7.*frame_index"):
        sample_problem_validation_state(
            ProblemValidationDemoConfig(dataset_root=root),
            np.random.default_rng(0),
        )


def test_sampler_rejects_missing_prefix_frame_index(tmp_path):
    root = _demo_dataset(tmp_path)
    _write_episode(
        root,
        7,
        [[float(frame)] * 7 for frame in range(10)],
        frame_indices=[0, 2, *range(3, 11)],
    )

    with pytest.raises(ValueError, match=r"Episode 7.*frame_index"):
        sample_problem_validation_state(
            ProblemValidationDemoConfig(dataset_root=root),
            np.random.default_rng(0),
        )


@pytest.mark.parametrize(
    "frame_indices",
    [
        ["invalid", *map(str, range(1, 10))],
        [0.5, *range(1, 10)],
        [-1, *range(1, 10)],
    ],
)
def test_sampler_rejects_invalid_frame_index(tmp_path, frame_indices):
    root = _demo_dataset(tmp_path)
    _write_episode(
        root,
        7,
        [[float(frame)] * 7 for frame in range(10)],
        frame_indices=frame_indices,
    )

    with pytest.raises(ValueError, match=r"Episode 7.*frame_index"):
        sample_problem_validation_state(
            ProblemValidationDemoConfig(dataset_root=root),
            np.random.default_rng(0),
        )


def test_prompt_controller_latches_success_and_runs_five_more_seconds():
    controller = PromptRunController(start_time=2.0, time_limit=30.0, post_success_seconds=5.0)

    assert controller.should_continue(10.0)
    assert not controller.observe(10.0, success=True)
    assert controller.success_time == 10.0
    assert controller.should_continue(14.999)
    assert not controller.observe(12.0, success=False)
    assert controller.observe(15.0, success=False)
    assert not controller.should_continue(15.0)


def test_sequence_runs_tail_then_sample_restore_then_second_prompt(tmp_path):
    config = ProblemValidationDemoConfig(dataset_root=tmp_path)
    events = []
    sampled = object()

    def run_prompt(prompt, post_success_seconds):
        events.append(("prompt", prompt, post_success_seconds))
        return True, True

    result = execute_problem_validation_sequence(
        config=config,
        rng=np.random.default_rng(3),
        run_prompt=run_prompt,
        sample_state=lambda _config, _rng: events.append(("sample",)) or sampled,
        restore_state=lambda state, steps: events.append(("restore", state, steps)),
    )

    assert result.success
    assert events == [
        ("prompt", config.prompts[0], 5.0),
        ("sample",),
        ("restore", sampled, 250),
        ("prompt", config.prompts[1], 0.0),
    ]


def test_sequence_stops_before_sampling_when_first_prompt_fails(tmp_path):
    config = ProblemValidationDemoConfig(dataset_root=tmp_path)
    events = []

    result = execute_problem_validation_sequence(
        config=config,
        rng=np.random.default_rng(3),
        run_prompt=lambda prompt, tail: (events.append((prompt, tail)) or (True, False)),
        sample_state=lambda *_: pytest.fail("must not sample"),
        restore_state=lambda *_: pytest.fail("must not restore"),
    )

    assert not result.success
    assert result.first_prompt_success is False
    assert result.second_prompt_success is None
