import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOBIO_ROOT = ROOT / "scripts" / "autobio_scripts"
if str(AUTOBIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOBIO_ROOT))

import centrifuge5910_tasks as centrifuge_tasks


def _reset_task(task_name: str, monkeypatch, random_choice: int):
    monkeypatch.setattr(
        centrifuge_tasks.np.random,
        "randint",
        lambda *args, **kwargs: random_choice,
    )
    task = centrifuge_tasks.Centrifuge5910Manipulate(
        centrifuge_tasks.Centrifuge5910Manipulate.load()
    )
    task.task = task_name
    task.reset(seed=17)
    return task


def _assert_tube_position(task, tube, expected_position):
    np.testing.assert_allclose(
        task._tube_position(tube),
        np.asarray(expected_position, dtype=np.float64),
        atol=1e-9,
    )


@pytest.mark.parametrize("random_choice", [0, 1])
def test_take_experimental_uses_slot_zero_and_randomizes_balance(
    monkeypatch,
    random_choice,
):
    task = _reset_task(
        "take_experimental_tube_from_centrifuge5910",
        monkeypatch,
        random_choice,
    )

    _assert_tube_position(task, task.tube, task._slot_position(0))
    if random_choice == 0:
        expected_balance_position = task._slot_position(1)
    else:
        expected_balance_position = task.rack1.get_position(
            task.data, 0, 2, "50ml"
        )
    _assert_tube_position(task, task.tube2, expected_balance_position)


@pytest.mark.parametrize("random_choice", [0, 1])
def test_take_balance_uses_slot_one_and_randomizes_experimental(
    monkeypatch,
    random_choice,
):
    task = _reset_task(
        "take_balance_tube_from_centrifuge5910",
        monkeypatch,
        random_choice,
    )

    _assert_tube_position(task, task.tube2, task._slot_position(1))
    if random_choice == 0:
        expected_experimental_position = task._slot_position(0)
    else:
        expected_experimental_position = task.rack1.get_position(
            task.data, 1, 4, "50ml"
        )
    _assert_tube_position(task, task.tube, expected_experimental_position)


def _expert_case_source(case_name: str) -> str:
    source = Path(centrifuge_tasks.__file__).read_text(encoding="utf-8")
    execute_source = source.split(
        "class Centrifuge5910ManipulateExpert", 1
    )[1].split("Centrifuge5910Manipulate.Expert =", 1)[0]
    start_marker = f"case '{case_name}':"
    branch = execute_source.split(start_marker, 1)[1]
    return branch.split("\n            case ", 1)[0]


def test_take_experimental_expert_targets_experimental_slot_and_tube():
    branch = _expert_case_source(
        "take_experimental_tube_from_centrifuge5910"
    )

    assert "slot_id = EXPERIMENTAL_TUBE_SLOT_ID" in branch
    assert "eef_pose = self.tube.get_end_effector_pose(self.data)" in branch


def test_take_balance_expert_targets_balance_slot_and_tube():
    branch = _expert_case_source("take_balance_tube_from_centrifuge5910")

    assert "slot_id = BALANCE_TUBE_SLOT_ID" in branch
    assert "eef_pose = self.tube2.get_end_effector_pose(self.data)" in branch
