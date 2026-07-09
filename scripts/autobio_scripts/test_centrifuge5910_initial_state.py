from pathlib import Path


CENTRIFUGE_SOURCE = Path("scripts/autobio_scripts/centrifuge5910_tasks.py")
THERMAL_CYCLER_SOURCE = Path("scripts/autobio_scripts/thermal_cycler_tasks.py")


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

    assert "lid_jntlimit[0]" in open_branch
    assert "lid_locked" in close_branch
    assert "_centrifuge5910_button_touched" in press_branch


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
