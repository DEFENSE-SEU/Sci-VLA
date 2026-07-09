import json
import sys
import types


def _import_transition_generation_with_stubs():
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = object
        sys.modules["openai"] = openai_stub
    import transition_generation

    return transition_generation


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
        max_plan_revisions=2,
        request_json_object=fake_request_json,
    )

    assert len(planning_prompts) == 2
    assert len(verifier_plans) == 2
    assert verification["passed"] is True
    assert commands[1]["distance_m"] == 0.1
    assert plan_obj["commands"] == commands


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


def test_workspace_bounds_are_in_verifier_prompt():
    transition_generation = _import_transition_generation_with_stubs()
    workspace = {
        "current_position_world": [0.1, 0.2, 0.3],
        "min_world": [-0.2, -0.3, 0.05],
        "max_world": [0.5, 0.6, 0.7],
        "margin_m": 0.03,
    }

    prompt = transition_generation._build_plan_verification_prompt(
        task_prompt="move to next task",
        planning_prompt="planner prompt",
        plan_obj={
            "commands": [{"op": "translate", "axis": "x", "distance_m": 0.1}],
            "plan_steps": ["translate x"],
            "final_target_qpos": [1, 2, 3, 4, 5, 6],
            "final_target_gripper": 0.0,
        },
        commands=[{"op": "translate", "axis": "x", "distance_m": 0.1}],
        ee_workspace_bounds=workspace,
    )

    assert "Conservative EE workspace boundary" in prompt
    assert '"current_position_world": [0.1, 0.2, 0.3]' in prompt
    assert "cumulative translate commands" in prompt


def test_reachability_checker_feedback_triggers_replanning():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    planning_prompts = []
    reachability_calls = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        content = kwargs["request_input"][0]["content"]
        prompt_text = content[0]["text"]

        if stage_name == "stage-1-planning":
            planning_prompts.append(prompt_text)
            if len(planning_prompts) == 1:
                return {
                    "commands": [
                        {"op": "translate", "axis": "x", "distance_m": 0.2, "steps": 50}
                    ],
                    "plan_steps": ["move x too far"],
                    "final_target_qpos": [0, 0, 0, 0, 0, 0],
                    "final_target_gripper": 0,
                }
            assert "Previous verifier feedback" in prompt_text
            assert "max_reachable_distance_m=0.12" in prompt_text
            return {
                "commands": [
                    {"op": "translate", "axis": "x", "distance_m": 0.1, "steps": 50}
                ],
                "plan_steps": ["move x within reach"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }

        if stage_name == "stage-1.5-plan-verifier":
            return {"passed": True, "issues": [], "bad_command_indices": [], "revision_instructions": []}

        raise AssertionError(f"unexpected stage: {stage_name}")

    def fake_reachability_checker(commands):
        reachability_calls.append(commands)
        if commands[0]["distance_m"] > 0.12:
            return {
                "passed": False,
                "issues": [
                    {
                        "command_index": 0,
                        "problem": (
                            "Requested translate x +0.2m is unreachable from cumulative EE pose; "
                            "max_reachable_distance_m=0.12."
                        ),
                        "required_fix": (
                            "Regenerate this command with abs(distance_m) <= 0.12 "
                            "or choose another axis/order."
                        ),
                    }
                ],
                "bad_command_indices": [0],
                "revision_instructions": [
                    "command_index=0 max_reachable_distance_m=0.12"
                ],
            }
        return {"passed": True, "issues": [], "bad_command_indices": [], "revision_instructions": []}

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
        ee_reachability_checker=fake_reachability_checker,
    )

    assert len(planning_prompts) == 2
    assert len(reachability_calls) == 2
    assert verification["passed"] is True
    assert commands[0]["distance_m"] == 0.1
    assert plan_obj["commands"] == commands


def test_planning_prompt_documents_directional_reachability_verifier():
    transition_generation = _import_transition_generation_with_stubs()
    _generate_verified_transition_plan = transition_generation._generate_verified_transition_plan

    captured_prompts = []

    def fake_request_json(**kwargs):
        stage_name = kwargs["stage_name"]
        content = kwargs["request_input"][0]["content"]
        prompt_text = content[0]["text"]
        if stage_name == "stage-1-planning":
            captured_prompts.append(prompt_text)
            return {
                "commands": [{"op": "translate", "axis": "z", "distance_m": 0.05}],
                "plan_steps": ["move z"],
                "final_target_qpos": [0, 0, 0, 0, 0, 0],
                "final_target_gripper": 0,
            }
        if stage_name == "stage-1.5-plan-verifier":
            return {"passed": True, "issues": [], "bad_command_indices": [], "revision_instructions": []}
        raise AssertionError(f"unexpected stage: {stage_name}")

    _generate_verified_transition_plan(
        client=object(),
        model_name="dummy-model",
        planning_prompt=(
            "Verifier will compute the max reachable distance for each translate "
            "axis/direction from the cumulative EE pose. Do not use recovery behavior."
        ),
        task_prompt="target task",
        front_image_data_url="front",
        side_image_data_url="side",
        target_front_image_data_url=None,
        max_plan_revisions=0,
        request_json_object=fake_request_json,
    )

    assert "max reachable distance for each translate axis/direction" in captured_prompts[0]
    assert "Do not use recovery behavior" in captured_prompts[0]


def test_local_workspace_verifier_catches_cumulative_translate_out_of_bounds():
    transition_generation = _import_transition_generation_with_stubs()
    workspace = {
        "current_position_world": [0.0, 0.0, 0.2],
        "min_world": [-0.3, -0.3, 0.0],
        "max_world": [0.3, 0.3, 0.6],
    }

    verification = transition_generation._verify_commands_with_ee_workspace_bounds(
        [
            {"op": "translate", "axis": "x", "distance_m": 0.25},
            {"op": "translate", "axis": "x", "distance_m": 0.1},
        ],
        workspace,
    )

    assert verification["passed"] is False
    assert verification["bad_command_indices"] == [1]
    assert "outside conservative EE workspace" in verification["issues"][0]["problem"]


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
