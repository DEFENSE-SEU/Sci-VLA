import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem_validation_demo import (
    ProblemValidationDemoConfig,
    PromptRunController,
    execute_problem_validation_sequence,
    sample_problem_validation_state,
)
from evaluate import apply_problem_validation_demo_profile, evaluate_task, parse_args


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
