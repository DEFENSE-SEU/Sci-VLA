import json
import math
import sys
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


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write_episode(root: Path, episode_index: int, states: list[list[float]]):
    path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    state_type = pa.list_(pa.float32(), 7)
    pq.write_table(
        pa.table(
            {
                "state": pa.array(states, type=state_type),
                "frame_index": pa.array(range(len(states)), type=pa.int64()),
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


def test_sampler_rejects_nonfinite_or_wrong_sized_state(tmp_path):
    root = _demo_dataset(tmp_path)
    _write_episode(root, 7, [[0.0] * 6 + [float("nan")]] * 10)

    with pytest.raises(ValueError, match="finite 7-dimensional"):
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
