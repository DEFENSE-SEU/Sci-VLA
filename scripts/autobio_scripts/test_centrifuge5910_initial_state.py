from pathlib import Path
import ast


CENTRIFUGE_SOURCE = Path("scripts/autobio_scripts/centrifuge5910_tasks.py")
THERMAL_CYCLER_SOURCE = Path("scripts/autobio_scripts/thermal_cycler_tasks.py")


CENTRIFUGE_ATOMIC_TASKS = {
    "open_centrifuge5910_lid",
    "place_experimental_tube_into_centrifuge5910",
    "place_balance_tube_into_centrifuge5910",
    "close_centrifuge5910_lid",
    "press_centrifuge5910_button",
    "take_experimental_tube_from_centrifuge5910",
    "take_balance_tube_from_centrifuge5910",
}

THERMAL_CYCLER_ATOMIC_TASKS = {
    "open_thermal_cycler_lid",
    "place_pcrPlate_into_thermalCycler",
    "close_thermal_cycler_lid",
    "screw_tighten_knob",
    "press_thermal_cycler_button",
    "screw_loosen_knob",
    "take_pcrPlate_from_thermalCycler",
}


def _literal_assignment(source: str, name: str):
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def _ranges_intersect(left, right):
    left_low, left_high = left
    right_low, right_high = right
    return all(
        left_low[index] <= right_high[index] and right_low[index] <= left_high[index]
        for index in range(len(left_low))
    )


def _assert_pairwise_disjoint(ranges):
    task_ranges = list(ranges.items())
    for left_index, (left_task, left_range) in enumerate(task_ranges):
        for right_task, right_range in task_ranges[left_index + 1:]:
            assert not _ranges_intersect(left_range, right_range), (
                f"{left_task} and {right_task} perturb ranges overlap"
            )


def test_long_task_2_uses_press_button_locked_lid_offset():
    source = CENTRIFUGE_SOURCE.read_text(encoding="utf-8")

    long_task_branch = source.split("case 'centrifuge5910_long_task_2':", 1)[1].split(
        "case 'open_centrifuge5910_lid':",
        1,
    )[0]
    press_button_branch = source.split("case 'press_centrifuge5910_button':", 1)[1].split(
        "case _:",
        1,
    )[0]

    expected = "self.instrument.lid_jntlimit[1] - 0.005"
    assert expected in press_button_branch
    assert expected in long_task_branch


def test_centrifuge5910_check_uses_requested_atomic_success_conditions():
    source = CENTRIFUGE_SOURCE.read_text(encoding="utf-8")
    check_body = source.split("def check(self, prompt: str | None = None):", 1)[1].split(
        "class Centrifuge5910ManipulateExpert",
        1,
    )[0]

    open_branch = check_body.split("case 'open_centrifuge5910_lid':", 1)[1].split("case ", 1)[0]
    close_branch = check_body.split("case 'close_centrifuge5910_lid'", 1)[1].split("case ", 1)[0]
    press_branch = check_body.split("case 'press_centrifuge5910_button':", 1)[1].split(
        "return False",
        1,
    )[0]
    take_experimental_branch = check_body.split(
        "case 'take_experimental_tube_from_centrifuge5910'",
        1,
    )[1].split("case ", 1)[0]
    take_balance_branch = check_body.split("case 'take_balance_tube_from_centrifuge5910':", 1)[
        1
    ].split("case ", 1)[0]

    assert "lid_jntlimit[0]" in open_branch
    assert "lid_locked" in close_branch
    assert "_centrifuge5910_button_touched" in press_branch
    assert "_tube_on_any_rack_slot(self.tube)" in take_experimental_branch
    assert "_tube_on_any_rack_slot(self.tube2)" in take_balance_branch


def test_thermal_cycler_check_uses_requested_atomic_success_conditions():
    source = THERMAL_CYCLER_SOURCE.read_text(encoding="utf-8")
    check_body = source.split("def check(self, prompt: str | None = None):", 1)[1].split(
        "class ThermalCyclerManipulateExpert",
        1,
    )[0]

    close_branch = check_body.split("case 'close_thermal_cycler_lid':", 1)[1].split("case ", 1)[0]
    press_branch = check_body.split("case 'press_thermal_cycler_button':", 1)[1].split(
        "return False",
        1,
    )[0]

    assert "lid_jntlimit[0]" in close_branch
    assert "lever_jntlimit[0]" in close_branch
    assert "_thermal_cycler_button_touched" in press_branch


def test_thermal_cycler_knob_success_checks_state_not_exact_target():
    source = THERMAL_CYCLER_SOURCE.read_text(encoding="utf-8")
    check_body = source.split("def check(self, prompt: str | None = None):", 1)[1].split(
        "class ThermalCyclerManipulateExpert",
        1,
    )[0]

    tighten_branch = check_body.split("case 'screw_tighten_knob':", 1)[1].split("case ", 1)[0]
    loosen_branch = check_body.split("case 'screw_loosen_knob':", 1)[1].split("case ", 1)[0]

    assert "_knob_tightened()" in tighten_branch
    assert "_knob_loosened()" in loosen_branch


def test_atomic_tasks_have_task_specific_arm_perturb_ranges():
    centrifuge_ranges = _literal_assignment(
        CENTRIFUGE_SOURCE.read_text(encoding="utf-8"),
        "CENTRIFUGE5910_ARM_PERTURB_RANGES",
    )
    thermal_ranges = _literal_assignment(
        THERMAL_CYCLER_SOURCE.read_text(encoding="utf-8"),
        "THERMAL_CYCLER_ARM_PERTURB_RANGES",
    )

    assert set(centrifuge_ranges) == CENTRIFUGE_ATOMIC_TASKS
    assert set(thermal_ranges) == THERMAL_CYCLER_ATOMIC_TASKS
    assert len(set(centrifuge_ranges.values())) == len(CENTRIFUGE_ATOMIC_TASKS)
    assert len(set(thermal_ranges.values())) == len(THERMAL_CYCLER_ATOMIC_TASKS)
    _assert_pairwise_disjoint(centrifuge_ranges)
    _assert_pairwise_disjoint(thermal_ranges)


def test_dataset_generation_prints_atomic_task_success():
    centrifuge_source = CENTRIFUGE_SOURCE.read_text(encoding="utf-8")
    thermal_source = THERMAL_CYCLER_SOURCE.read_text(encoding="utf-8")

    assert "success = bool(expert.check())" in centrifuge_source
    assert "print(f\"task={task} episode={i:04d} attempt={attempt} success={success}\")" in centrifuge_source
    assert "success = bool(expert.check())" in thermal_source
    assert "print(f\"task={task} episode={i:04d} attempt={attempt} success={success}\")" in thermal_source


def test_failed_dataset_generation_removes_failed_episode_and_retries():
    centrifuge_source = CENTRIFUGE_SOURCE.read_text(encoding="utf-8")
    thermal_source = THERMAL_CYCLER_SOURCE.read_text(encoding="utf-8")

    for source in (centrifuge_source, thermal_source):
        assert "import shutil" in source
        assert "max_attempts_per_episode" in source
        assert "generated_dir = Path(expert.serializer.save_dir)" in source
        assert "failed_dir = generated_dir" in source
        assert "failed_dir = Path(expert.serializer.save_dir)" not in source
        assert "shutil.rmtree(failed_dir)" in source
        assert "episode={i:04d} attempt={attempt}" in source
        assert "raise RuntimeError" in source
