import json
from pathlib import Path
import sys
import types


def _import_transition_generation_with_stubs():
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = object
        sys.modules["openai"] = openai_stub
    import transition_generation

    return transition_generation


def test_verified_planning_skips_llm_verifier_by_default():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    requested_stages = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        requested_stages.append(stage_name)
        if stage_name == "stage-1-planning":
            return {
                "commands": [{"op": "translate", "axis": "z", "distance_m": 0.1}],
                "plan_steps": ["translate upward"],
                "safety_notes": ["transition only"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }
        if stage_name == "stage-1.5-plan-verifier":
            raise AssertionError("verifier should be disabled by default")
        raise AssertionError(f"unexpected stage: {stage_name}")

    plan_obj, commands, verification = _generate_verified_transition_plan(
        client=object(),
        model_name="dummy-model",
        planning_prompt="planner prompt",
        task_prompt="target task",
        front_image_data_url="front",
        side_image_data_url="side",
        target_front_image_data_url=None,
        request_json_object=fake_request_json,
    )

    assert requested_stages == ["stage-1-planning"]
    assert verification == {
        "passed": True,
        "issues": [],
        "bad_command_indices": [],
        "revision_instructions": [],
        "verifier_enabled": False,
    }
    assert commands == [{"op": "translate", "axis": "z", "distance_m": 0.1}]
    assert plan_obj["commands"] == commands



def test_disabled_verifier_accepts_over_limit_translate_without_plan_constraint_check():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    requested_stages = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        requested_stages.append(stage_name)
        if stage_name == "stage-1-planning":
            return {
                "commands": [{"op": "translate", "axis": "z", "distance_m": 0.2711}],
                "plan_steps": ["translate upward"],
                "safety_notes": ["verifier disabled regression"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }
        if stage_name == "stage-1.5-plan-verifier":
            raise AssertionError("verifier should be disabled")
        raise AssertionError(f"unexpected stage: {stage_name}")

    plan_obj, commands, verification = _generate_verified_transition_plan(
        client=object(),
        model_name="dummy-model",
        planning_prompt="planner prompt",
        task_prompt="target task",
        front_image_data_url="front",
        side_image_data_url="side",
        target_front_image_data_url=None,
        verifier_enabled=False,
        request_json_object=fake_request_json,
    )

    assert requested_stages == ["stage-1-planning"]
    assert verification["passed"] is True
    assert verification["verifier_enabled"] is False
    assert commands == [{"op": "translate", "axis": "z", "distance_m": 0.2711}]
    assert plan_obj["commands"] == commands
    execute_body = transition_generation._commands_to_execute_body(
        commands,
        enforce_plan_constraints=False,
    )
    assert "0.2711" in execute_body


def test_verified_planning_retries_with_bad_action_feedback():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    planning_prompts = []
    verifier_plans = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        content = kwargs["request_input"][0]["content"]
        prompt_text = content[0]["text"]

        if stage_name == "stage-1-planning":
            planning_prompts.append(prompt_text)
            if len(planning_prompts) == 1:
                return {
                    "commands": [
                        {"op": "open_gripper", "delay": 100},
                        {"op": "translate", "axis": "z", "distance_m": 0.1, "steps": 50},
                    ],
                    "plan_steps": ["open gripper", "rotate around x"],
                    "safety_notes": ["constraint test"],
                    "final_target_qpos": [0, 0, 0, 0, 0, 0],
                    "final_target_gripper": 0,
                }

            assert "Previous verifier feedback" in prompt_text
            assert "command_index=1" in prompt_text
            assert "rotate around x" in prompt_text
            return {
                "commands": [
                    {"op": "open_gripper", "delay": 100},
                    {"op": "translate", "axis": "z", "distance_m": 0.1, "steps": 50},
                ],
                "plan_steps": ["open gripper", "translate along z"],
                "safety_notes": ["clear upward first"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }

        if stage_name == "stage-1.5-plan-verifier":
            plan_json = prompt_text.split("Plan JSON to verify:", 1)[1].strip()
            verifier_plans.append(json.loads(plan_json))
            if len(verifier_plans) == 1:
                return {
                    "passed": False,
                    "issues": [
                        {
                            "command_index": 1,
                            "problem": "plan_steps entry 1 does not match command 1.",
                            "required_fix": "Make plan_steps entry 1 describe the translate command.",
                        }
                    ],
                    "bad_command_indices": [1],
                    "revision_instructions": ["Update plan_steps entry 1 to match command 1."],
                }
            return {"passed": True, "issues": [], "bad_command_indices": [], "revision_instructions": []}

        raise AssertionError(f"unexpected stage: {stage_name}")

    plan_obj, commands, verification = _generate_verified_transition_plan(
        client=object(),
        model_name="dummy-model",
        planning_prompt="planner prompt",
        task_prompt="target task",
        front_image_data_url="front",
        side_image_data_url="side",
        target_front_image_data_url=None,
        verifier_enabled=True,
        max_plan_revisions=2,
        request_json_object=fake_request_json,
    )

    assert len(planning_prompts) == 2
    assert len(verifier_plans) == 2
    assert verification["passed"] is True
    assert commands[1]["distance_m"] == 0.1
    assert plan_obj["commands"] == commands


def test_replace_execute_body_uses_rrt_for_final_target_restore():
    transition_generation = _import_transition_generation_with_stubs()

    template_code = """
class TransitionExpert:
    def __init__(self):
        pass

    def execute(self):
        pass
""".strip()

    code = transition_generation._replace_execute_body(
        template_code,
        execute_body_code='self.execute_transition_commands([{"op": "wait", "steps": 1}])',
        final_target_qpos=[1, 2, 3, 4, 5, 6],
        final_target_gripper=0.0,
        include_final_restore=True,
    )

    assert "self.move_to_target_qpos_rrt(target_qpos)" in code
    assert "self.move_to_target_qpos(target_qpos)" not in code


def test_replace_execute_body_uses_rrt_validation_for_target_candidates():
    transition_generation = _import_transition_generation_with_stubs()

    template_code = """
class TransitionExpert:
    def __init__(self):
        pass

    def execute(self):
        pass
""".strip()

    code = transition_generation._replace_execute_body(
        template_code,
        execute_body_code='self.execute_transition_commands([{"op": "wait", "steps": 1}])',
        final_target_qpos=[1, 2, 3, 4, 5, 6],
        final_target_gripper=0.0,
        include_final_restore=True,
        final_target_qpos_candidates=[[1, 2, 3, 4, 5, 6, 0]],
    )

    assert "validate_qpos_rrt_path" in code
    assert "validate_qpos_interpolation_path" not in code


def test_validate_code_rejects_direct_motion_in_execute_body():
    transition_generation = _import_transition_generation_with_stubs()

    code = """
class TransitionExpert:
    def move_to(self, pose):
        pass

    def execute(self):
        self.move_to(None)
""".strip()

    is_valid, validation_msg = transition_generation.validate_code(code)

    assert is_valid is False
    assert "Direct non-RRT motion call in execute" in validation_msg


def test_no_retrieval_without_final_restore_does_not_require_final_target_qpos():
    transition_generation = _import_transition_generation_with_stubs()

    plan_target_qpos, plan_target_gripper = transition_generation._resolve_plan_restore_targets(
        plan_obj={
            "commands": [{"op": "translate", "axis": "z", "distance_m": 0.1}],
            "plan_steps": ["translate upward"],
            "final_target_qpos": None,
            "final_target_gripper": 255,
        },
        target_arm_qpos=[0, 1, 2, 3, 4, 5],
        target_gripper_state=None,
        include_final_restore=False,
    )

    assert plan_target_qpos == [0, 1, 2, 3, 4, 5]
    assert plan_target_gripper is None


def test_no_retrieval_restore_schema_does_not_request_target_qpos():
    transition_generation = _import_transition_generation_with_stubs()

    schema_text = transition_generation._format_restore_schema_fields(
        no_retrieval=True,
        target_arm_qpos=[0, 1, 2, 3, 4, 5],
        target_gripper_state=None,
    )

    assert '"restore": false' in schema_text
    assert "final_target_qpos" not in schema_text
    assert "final_target_gripper" not in schema_text


def test_null_final_target_qpos_still_fails_when_final_restore_is_required():
    transition_generation = _import_transition_generation_with_stubs()

    try:
        transition_generation._resolve_plan_restore_targets(
            plan_obj={
                "commands": [{"op": "translate", "axis": "z", "distance_m": 0.1}],
                "plan_steps": ["translate upward"],
                "final_target_qpos": None,
            },
            target_arm_qpos=[0, 1, 2, 3, 4, 5],
            target_gripper_state=None,
            include_final_restore=True,
        )
    except ValueError as exc:
        assert "Stage-1 planning output missing valid final_target_qpos" in str(exc)
    else:
        raise AssertionError("expected null final_target_qpos to fail when final restore is required")


def test_verified_planning_raises_after_revision_limit():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        if stage_name == "stage-1-planning":
            return {
                "commands": [{"op": "translate", "axis": "z", "distance_m": -0.1}],
                "plan_steps": ["move down"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }
        if stage_name == "stage-1.5-plan-verifier":
            return {
                "passed": False,
                "issues": [
                    {
                        "command_index": 0,
                        "problem": "plan_steps entry 0 does not match command 0.",
                        "required_fix": "Update plan_steps entry 0 to match command 0.",
                    }
                ],
                "bad_command_indices": [0],
                "revision_instructions": ["Update plan_steps entry 0 to match command 0."],
            }
        raise AssertionError(f"unexpected stage: {stage_name}")

    try:
        _generate_verified_transition_plan(
            client=object(),
            model_name="dummy-model",
            planning_prompt="planner prompt",
            task_prompt="target task",
            front_image_data_url="front",
            side_image_data_url="side",
            target_front_image_data_url=None,
            verifier_enabled=True,
            max_plan_revisions=1,
            request_json_object=fake_request_json,
        )
    except ValueError as exc:
        assert "Plan verification failed after 1 revision" in str(exc)
        assert "command_index=0" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_verifier_prompt_documents_gripper_and_restore_target_semantics():
    transition_generation = _import_transition_generation_with_stubs()
    prompt = transition_generation._build_plan_verification_prompt(
        task_prompt="open the lid",
        planning_prompt="planner prompt",
        plan_obj={
            "commands": [{"op": "open_gripper"}],
            "plan_steps": ["open gripper"],
            "final_target_qpos": [1, 2, 3, 4, 5, 6],
            "final_target_gripper": 0.0,
        },
        commands=[{"op": "open_gripper"}],
    )

    assert "0 means fully open" in prompt
    assert "255 means fully closed" in prompt
    assert "not a normalized 0/1 flag" in prompt
    assert "final_target_qpos and final_target_gripper are host restore targets" in prompt
    assert "must not be forced to match the last transition command" in prompt


def test_verifier_result_filters_known_invalid_restore_and_gripper_issues():
    transition_generation = _import_transition_generation_with_stubs()

    result = transition_generation._normalize_plan_verification_result(
        {
            "passed": False,
            "issues": [
                {
                    "command_index": 9,
                    "problem": "final_target_gripper 0.0 contradicts open_gripper because 0.0 implies closed",
                    "required_fix": "Update final_target_gripper to 1.0 (Open)",
                },
                {
                    "command_index": 5,
                    "problem": "final_target_qpos matches approach pose, not the cumulative motion after retracting",
                    "required_fix": "Update final_target_qpos to reflect cumulative motion.",
                },
            ],
            "bad_command_indices": [9, 5],
            "revision_instructions": [
                "Correct final_target_gripper to 1.0 (Open)",
                "Update final_target_qpos to reflect cumulative motion.",
            ],
        },
        command_count=10,
    )

    assert result["passed"] is True
    assert result["issues"] == []
    assert result["bad_command_indices"] == []
    assert result["revision_instructions"] == []


def test_verifier_prompt_is_constraint_only_not_action_reasonableness():
    transition_generation = _import_transition_generation_with_stubs()
    prompt = transition_generation._build_plan_verification_prompt(
        task_prompt="open the lid",
        planning_prompt="planner prompt",
        plan_obj={
            "commands": [{"op": "close_gripper"}],
            "plan_steps": ["close gripper"],
            "final_target_qpos": [1, 2, 3, 4, 5, 6],
            "final_target_gripper": 0.0,
        },
        commands=[{"op": "close_gripper"}],
    )

    assert "constraint-only checker" in prompt
    assert "Do not judge whether an action choice is reasonable" in prompt
    assert "Do not reject a plan because a gripper action seems unsafe" in prompt


def test_transition_generation_source_has_no_legacy_workspace_or_reachability_agent_args():
    transition_generation = _import_transition_generation_with_stubs()
    source = Path("scripts/autobio_scripts/transition_generation.py").read_text(encoding="utf-8")

    assert "allowed_apis" not in source
    assert "_collect_execute_allowed_apis" not in source
    assert "ee_workspace_bounds" not in source
    assert "ee_reachability_checker" not in source
    assert "_verify_commands_with_ee_workspace_bounds" not in source
    assert "_format_ee_workspace_bounds_for_prompt" not in source
    assert "_normalize_ee_workspace_bounds" not in source
    assert "_merge_plan_verifications" not in source
    assert "ee_workspace_bounds" not in transition_generation._build_plan_verification_prompt.__annotations__


def test_generate_verified_transition_plan_no_longer_accepts_reachability_checker():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    planning_prompts = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        content = kwargs["request_input"][0]["content"]
        prompt_text = content[0]["text"]

        if stage_name == "stage-1-planning":
            planning_prompts.append(prompt_text)
            return {
                "commands": [
                    {"op": "translate", "axis": "x", "distance_m": 0.2, "steps": 50}
                ],
                "plan_steps": ["move x within single-command distance limit"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }

        if stage_name == "stage-1.5-plan-verifier":
            return {"passed": True, "issues": [], "bad_command_indices": [], "revision_instructions": []}

        raise AssertionError(f"unexpected stage: {stage_name}")

    plan_obj, commands, verification = _generate_verified_transition_plan(
        client=object(),
        model_name="dummy-model",
        planning_prompt="planner prompt",
        task_prompt="target task",
        front_image_data_url="front",
        side_image_data_url="side",
        target_front_image_data_url=None,
        max_plan_revisions=2,
        request_json_object=fake_request_json,
    )

    assert len(planning_prompts) == 1
    assert verification["passed"] is True
    assert commands[0]["distance_m"] == 0.2
    assert plan_obj["commands"] == commands


def test_planning_prompt_does_not_document_directional_reachability_verifier():
    transition_generation = _import_transition_generation_with_stubs()
    prompt_text = transition_generation._format_ee_reachability_for_prompt(True)

    assert "max reachable distance for each translate axis/direction" not in prompt_text
    assert "Do not use recovery behavior" not in prompt_text
    assert "abs(distance_m) <= 0.25m" in prompt_text


def test_transition_planning_prompt_limits_work_to_transition_not_next_task():
    transition_generation = _import_transition_generation_with_stubs()

    prompt_text = transition_generation._build_transition_planning_prompt(
        target_reference_text="the target task's retrieved initial front-view image.",
        target_binding_text="The third image is the TARGET INITIAL FRONT reference view.",
        calibration_prompt_text="calibration",
        spatial_context_prompt_text="spatial context",
        reachability_prompt_text="movement limits",
        restore_schema_fields='"final_target_qpos": [0, 0, 0, 0, 0, 0],\n    "final_target_gripper": 0.0',
        motion_constraint_rule="single-command limits only",
    )

    assert "retrieved starting pose for the next atomic task" in prompt_text
    assert "Do not execute the next atomic task" in prompt_text
    assert "opening or closing lids" in prompt_text
    assert "pressing buttons" in prompt_text
    assert "turning knobs" in prompt_text
    assert "then target approach" not in prompt_text


def test_verifier_result_ignores_action_reasonableness_issues_but_keeps_constraints():
    transition_generation = _import_transition_generation_with_stubs()

    result = transition_generation._normalize_plan_verification_result(
        {
            "passed": False,
            "issues": [
                {
                    "command_index": 6,
                    "problem": "Closing the gripper immediately after approaching the target may be unsafe.",
                    "required_fix": "Add a safety check before grasping.",
                },
                {
                    "command_index": 2,
                    "problem": "translate distance_m 0.3 exceeds the 0.25m single-command limit.",
                    "required_fix": "Split the move into two translate commands.",
                },
            ],
            "bad_command_indices": [6, 2],
            "revision_instructions": [
                "Add a safety check before closing the gripper.",
                "Split command 2 because distance_m exceeds 0.25.",
            ],
        },
        command_count=8,
    )

    assert result["passed"] is False
    assert result["bad_command_indices"] == [2]
    assert len(result["issues"]) == 1
    assert result["issues"][0]["command_index"] == 2
    assert result["revision_instructions"] == ["Split command 2 because distance_m exceeds 0.25."]


def test_verifier_result_keeps_next_task_semantic_action_issues():
    transition_generation = _import_transition_generation_with_stubs()

    result = transition_generation._normalize_plan_verification_result(
        {
            "passed": False,
            "issues": [
                {
                    "command_index": 3,
                    "problem": "Command 3 executes the next atomic task by pressing buttons instead of only doing transition.",
                    "required_fix": "Remove next-task semantic task action and keep transition-only movement.",
                }
            ],
            "bad_command_indices": [3],
            "revision_instructions": [
                "Do not execute the next atomic task; remove semantic task action from command 3."
            ],
        },
        command_count=5,
    )

    assert result["passed"] is False
    assert result["bad_command_indices"] == [3]
    assert len(result["issues"]) == 1
    assert "next atomic task" in result["issues"][0]["problem"]
    assert result["revision_instructions"] == [
        "Do not execute the next atomic task; remove semantic task action from command 3."
    ]
