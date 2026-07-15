# Problem Validation Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command evaluate mode that records the fixed open-lid → five-second VLA tail → sampled early placement-state interpolation → place-PCR-plate workflow.

**Architecture:** A new pure helper module owns the immutable demo profile, deterministic LeRobot state sampling, prompt tail timing, and callback-based phase ordering. `evaluate.py` applies the fixed CLI profile, while `evaluator.py` adapts the existing policy loop, task predicates, interpolation, and replay recorder to that helper without changing normal evaluation behavior.

**Tech Stack:** Python 3.11, NumPy, PyArrow/Parquet, MuJoCo, imageio, pytest, existing WebSocket policy client.

## Global Constraints

- The flow is fixed to `thermal_cycler_long_task_1` with prompts `open the lid of the thermal cycler` then `place pcrPlate into the thermal cycler`.
- After the first local success is first observed, VLA inference and action execution continue for at least five additional seconds of simulation time before transition, with overshoot bounded by one executed control step.
- Select one placement trajectory randomly, then select one state uniformly from frames `0..ceil(0.30 * length)-1`; both selections use the evaluation seed.
- Restore all seven state values—six arm joints and one gripper—by direct interpolation only.
- Preserve non-robot MuJoCo state and capture one uninterrupted replay across both prompts and interpolation.
- Existing evaluation modes and user-owned dirty-worktree changes must remain untouched.

---

## File Structure

- Create `scripts/autobio_scripts/problem_validation_demo.py`: fixed config, seeded dataset sampler, post-success timing controller, and callback-based phase sequence.
- Create `scripts/autobio_scripts/test_problem_validation_demo.py`: unit tests for sampling, timing, ordering, interpolation wiring, and CLI profile behavior.
- Modify `scripts/autobio_scripts/evaluate.py`: expose/apply the one-command option and pass the demo config into serial evaluation.
- Modify `scripts/autobio_scripts/evaluator.py`: add seven-dimensional restore and adapt the existing policy/replay loop to the fixed sequence.
- Modify `README.md`: document the policy-server and demo commands plus output naming.

### Task 1: Deterministic Demo Domain Module

**Files:**
- Create: `scripts/autobio_scripts/problem_validation_demo.py`
- Create: `scripts/autobio_scripts/test_problem_validation_demo.py`

**Interfaces:**
- Produces: `ProblemValidationDemoConfig`, `SampledRobotState`, `PromptRunController`, `sample_problem_validation_state(config, rng)`, and `execute_problem_validation_sequence(...)`.
- Consumes: local LeRobot v2.1 metadata and episode Parquet files.

- [ ] **Step 1: Write failing sampler and timing tests**

Create tests that build a two-episode miniature dataset and assert exact filtering, prefix bounds, reproducibility, validation, and the latched five-second tail:

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q scripts/autobio_scripts/test_problem_validation_demo.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'problem_validation_demo'`.

- [ ] **Step 3: Implement config, sampler, and timing controller**

Implement the module with these public shapes and behavior:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProblemValidationDemoConfig:
    task_name: str = "thermal_cycler_long_task_1"
    prompts: tuple[str, str] = (
        "open the lid of the thermal cycler",
        "place pcrPlate into the thermal cycler",
    )
    post_success_seconds: float = 5.0
    dataset_repo_id: str = "mani_thermalcycler"
    dataset_root: Path | None = None
    prefix_fraction: float = 0.30
    interpolation_steps: int = 250
    video_filename_prefix: str = "problem_validation_open_lid_place_pcr_plate"


@dataclass(frozen=True, eq=False)
class SampledRobotState:
    state: np.ndarray
    episode_index: int
    episode_length: int
    prefix_frame_count: int
    frame_index: int
    frame_ratio: float
    task: str
    dataset_root: Path

    def __eq__(self, other):
        return (
            isinstance(other, SampledRobotState)
            and np.array_equal(self.state, other.state)
            and self.episode_index == other.episode_index
            and self.episode_length == other.episode_length
            and self.prefix_frame_count == other.prefix_frame_count
            and self.frame_index == other.frame_index
            and self.task == other.task
            and self.dataset_root == other.dataset_root
        )


@dataclass
class PromptRunController:
    start_time: float
    time_limit: float
    post_success_seconds: float = 0.0
    success_time: float | None = None

    def should_continue(self, current_time: float) -> bool:
        if self.success_time is None:
            return current_time - self.start_time < self.time_limit
        return current_time - self.success_time < self.post_success_seconds

    def observe(self, current_time: float, *, success: bool | None) -> bool:
        if success is True and self.success_time is None:
            self.success_time = float(current_time)
        return self.success_time is not None and not self.should_continue(current_time)

    @property
    def succeeded(self) -> bool:
        return self.success_time is not None
```

`sample_problem_validation_state` must default `dataset_root` to `Path.home() / ".cache/huggingface/lerobot" / config.dataset_repo_id`, parse JSONL defensively, resolve the `data_path` template using `episode_chunk = episode_index // chunks_size`, import `pyarrow.parquet` lazily, read only `state` and `frame_index`, and reject missing, empty, non-finite, or non-seven-dimensional data with contextual `ValueError`/`FileNotFoundError` messages.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all three tests pass.

- [ ] **Step 5: Write failing sequence-order tests**

Append tests for the callback state machine:

```python
from problem_validation_demo import execute_problem_validation_sequence


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
```

- [ ] **Step 6: Verify RED, implement sequence, verify GREEN**

Run the focused tests and observe an import failure for `execute_problem_validation_sequence`. Add `ProblemValidationDemoResult` and the callback function; it must return immediately on unhealthy/failed first prompt, propagate sampler/restore exceptions, and run the second prompt only after restoration. Re-run and expect all tests to pass.

- [ ] **Step 7: Commit the domain module**

```bash
git add scripts/autobio_scripts/problem_validation_demo.py scripts/autobio_scripts/test_problem_validation_demo.py
git commit -m "feat: add problem validation demo state machine"
```

### Task 2: One-Command Evaluate Profile

**Files:**
- Modify: `scripts/autobio_scripts/evaluate.py`
- Test: `scripts/autobio_scripts/test_problem_validation_demo.py`

**Interfaces:**
- Consumes: `ProblemValidationDemoConfig` from Task 1.
- Produces: `apply_problem_validation_demo_profile(args) -> ProblemValidationDemoConfig | None` and `--problem-validation-demo`.

- [ ] **Step 1: Write the failing CLI profile test**

```python
from argparse import Namespace
from evaluate import apply_problem_validation_demo_profile


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
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q scripts/autobio_scripts/test_problem_validation_demo.py::test_demo_profile_overrides_conflicting_generic_evaluate_args
```

Expected: import failure because the resolver is absent.

- [ ] **Step 3: Implement and wire the CLI profile**

Add the Boolean flag in `parse_args`, implement the resolver exactly once, call it immediately after parsing, print its effective values, and pass the returned config through `evaluate_task` into `Evaluator.evaluate` in the serial path. Leave worker globals and initialization unchanged because the fixed profile forces `num_workers=0`. Add the optional parameter at the end of function signatures to preserve positional compatibility.

- [ ] **Step 4: Verify profile and existing evaluate tests**

```bash
pytest -q scripts/autobio_scripts/test_problem_validation_demo.py scripts/autobio_scripts/test_evaluate_atomic_task_summary.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit evaluate wiring**

```bash
git add scripts/autobio_scripts/evaluate.py scripts/autobio_scripts/test_problem_validation_demo.py
git commit -m "feat: add fixed problem validation evaluate option"
```

### Task 3: Evaluator Adapter, Seven-Dimensional Restore, and Continuous Replay

**Files:**
- Modify: `scripts/autobio_scripts/evaluator.py`
- Test: `scripts/autobio_scripts/test_problem_validation_demo.py`

**Interfaces:**
- Consumes: `ProblemValidationDemoConfig`, `PromptRunController`, `execute_problem_validation_sequence`, `sample_problem_validation_state`.
- Produces: `restore_robot_state_direct(...) -> int` and a demo-only adapter inside `Evaluator.evaluate`.

- [ ] **Step 1: Write the failing seven-dimensional interpolation test**

```python
from evaluator import restore_robot_state_direct


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
```

- [ ] **Step 2: Verify RED, implement restore, verify GREEN**

Run the single test and observe the missing import. Implement validation plus linear controls for all seven action indices, then re-run. Keep `restore_prompt_initial_state_direct` working by delegating its resolved target to the new helper.

- [ ] **Step 3: Adapt `run_prompt` to `PromptRunController`**

Change the nested function signature to:

```python
def run_prompt(
    prompt: str | None,
    current_time_limit: float,
    post_success_seconds: float = 0.0,
):
```

Use the controller so ordinary prompts still stop immediately at success, while the first demo prompt latches success and continues policy calls/actions for five simulation seconds. Print `[ProblemValidationDemo] first_success_time=...` once and a tail-complete diagnostic once.

- [ ] **Step 4: Add the fixed sequence adapter before generic multi-prompt handling**

When `problem_validation_demo_config` is present:

- assert the two effective prompts match the config;
- call the pure sequence with a `run_prompt` callback using the existing time limit;
- wrap restoration `task.step_and_log` so collision counting and `_capture_replay_frame()` continue during all 250 interpolation steps;
- append one `atomic_task_results` record per attempted prompt;
- log the selected episode, length, prefix size, frame, ratio, state, and restore step count;
- use `try/except/finally` so `task.finish`, `render_finish`, and `save_video(..., filename_override=config.video_filename_prefix)` execute exactly once;
- save failure video before re-raising sampling/restoration errors;
- return the sequence success and existing timing statistics.

- [ ] **Step 5: Run focused and surrounding evaluator tests**

```bash
pytest -q \
  scripts/autobio_scripts/test_problem_validation_demo.py \
  scripts/autobio_scripts/test_non_llm_transition.py \
  scripts/autobio_scripts/test_evaluate_atomic_task_summary.py \
  scripts/autobio_scripts/test_evaluator_reachability_source.py \
  scripts/autobio_scripts/test_transition_collision_metrics.py
```

Expected: all pass with no new warnings or collection errors.

- [ ] **Step 6: Commit evaluator integration**

```bash
git add scripts/autobio_scripts/evaluator.py scripts/autobio_scripts/test_problem_validation_demo.py
git commit -m "feat: run fixed two-stage VLA validation demo"
```

### Task 4: User Documentation and Static CLI Verification

**Files:**
- Modify: `README.md`
- Test: `scripts/autobio_scripts/test_problem_validation_demo.py`

**Interfaces:**
- Documents: policy server prerequisite, one-command demo, seed behavior, dataset path, and MP4 names.

- [ ] **Step 1: Add a failing documentation/CLI smoke assertion**

Add a test that reads `README.md` and asserts it contains `--problem-validation-demo`, both fixed prompts, `mani_thermalcycler`, and the `problem_validation_open_lid_place_pcr_plate` output prefix.

- [ ] **Step 2: Verify RED, update README, verify GREEN**

Add a “Problem-validation video demo” subsection under Evaluation with this invocation:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --problem-validation-demo \
  --seed 0 \
  --time_limit 30
```

Explain that the policy server must already be running, the local dataset must exist at `~/.cache/huggingface/lerobot/mani_thermalcycler`, seed controls trajectory/frame sampling, and front/left MP4s are saved under `videos/` with the fixed prefix.

- [ ] **Step 3: Verify CLI help without starting MuJoCo evaluation**

```bash
python scripts/autobio_scripts/evaluate.py --help | rg "problem-validation-demo"
```

Expected: one matching option/help line.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md scripts/autobio_scripts/test_problem_validation_demo.py
git commit -m "docs: explain problem validation video demo"
```

### Task 5: Full Verification and Requested MP4

**Files:**
- Verify only; generated MP4s remain untracked under `videos/`.

**Interfaces:**
- Requires: a compatible OpenPI/LABVLA WebSocket policy server at the selected host/port and the local `mani_thermalcycler` dataset.

- [ ] **Step 1: Run syntax and focused regression checks**

```bash
python -m py_compile \
  scripts/autobio_scripts/problem_validation_demo.py \
  scripts/autobio_scripts/evaluate.py \
  scripts/autobio_scripts/evaluator.py

pytest -q \
  scripts/autobio_scripts/test_problem_validation_demo.py \
  scripts/autobio_scripts/test_non_llm_transition.py \
  scripts/autobio_scripts/test_evaluate_atomic_task_summary.py \
  scripts/autobio_scripts/test_evaluator_reachability_source.py \
  scripts/autobio_scripts/test_transition_collision_metrics.py
```

Expected: compilation succeeds and all selected tests pass.

- [ ] **Step 2: Confirm the dataset and policy endpoint**

```bash
test -f /home/evan/.cache/huggingface/lerobot/mani_thermalcycler/meta/info.json
ss -ltn | rg ':8000\b'
```

Expected: dataset test returns zero and the policy port is listening. If the endpoint is not listening, start the policy server using the README checkpoint command before continuing.

- [ ] **Step 3: Generate the requested video**

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --problem-validation-demo \
  --seed 0 \
  --time_limit 30 \
  --host 127.0.0.1 \
  --port 8000
```

Expected: logs show first success, five-second tail completion, a sampled placement episode/frame inside its first 30%, 250 restore steps, second-prompt result, and saved front/left MP4 paths.

- [ ] **Step 4: Validate the generated media**

```bash
find videos -maxdepth 1 -type f -name 'problem_validation_open_lid_place_pcr_plate*.mp4' -print
```

Use `ffprobe` or `imageio` to confirm each reported MP4 has a positive frame count and duration. Visually inspect representative frames before, during, and after interpolation if a local viewer is available.

- [ ] **Step 5: Review final diff and status**

```bash
git diff --check
git status --short
git log --oneline -6
```

Expected: no whitespace errors; only intended feature changes plus the user's pre-existing unrelated modifications are present; generated videos are not staged.

---

## Plan Self-Review

- Spec coverage: the fixed CLI, seeded first-30% sampling, five-second VLA tail, seven-dimensional interpolation, continuous replay, error finalization, docs, and real MP4 generation each map to a task above.
- Placeholder scan: no deferred implementation placeholders are used.
- Type consistency: the same `ProblemValidationDemoConfig`, `SampledRobotState`, `PromptRunController`, and callback signatures are used across all tasks.
