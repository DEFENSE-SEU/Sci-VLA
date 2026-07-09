from pathlib import Path


def test_evaluator_wires_directional_reachability_checker_into_codegen():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    assert "build_ur5e_directional_reachability_checker" in source
    assert "ee_reachability_checker = build_ur5e_directional_reachability_checker(" in source
    assert "ee_reachability_checker=ee_reachability_checker" in source


def test_evaluator_reachability_feedback_reports_max_distance():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    assert "max_reachable_distance_m" in source
    assert "Regenerate this command with abs(distance_m)" in source
    assert "Do not use recovery" in source
