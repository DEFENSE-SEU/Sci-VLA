from pathlib import Path


def test_evaluator_does_not_wire_directional_reachability_checker_into_codegen():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    assert "build_ur5e_directional_reachability_checker" not in source
    assert "ee_reachability_checker" not in source


def test_evaluator_reachability_feedback_is_not_used_for_planning():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    assert "max_reachable_distance_m" not in source
    assert "compute_ur5e_ee_workspace_static_bounds" not in source
    assert "attach_current_ee_position_to_workspace_bounds" not in source
    assert "_ur5e_ee_workspace_static_bounds" not in source
    assert "ee_workspace_bounds" not in source


def test_evaluator_reloads_generated_transition_template_before_execution():
    source = Path("scripts/autobio_scripts/evaluator.py").read_text(encoding="utf-8")

    generation_index = source.index("transition_code_generation(")
    invalidate_index = source.index("importlib.invalidate_caches()", generation_index)
    reload_index = source.index(
        "transition_template = importlib.reload(transition_template)",
        invalidate_index,
    )
    expert_index = source.index(
        "TransitionExpert = transition_template.TransitionExpert",
        reload_index,
    )

    assert generation_index < invalidate_index < reload_index < expert_index
    assert "from transition_template import TransitionExpert" not in source
