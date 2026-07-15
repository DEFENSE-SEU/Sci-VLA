# Problem Validation Demo RRT Restore Implementation Plan

**Goal:** Replace the fixed demo's direct robot-state interpolation with collision-aware six-joint RRT planning, followed by gripper restoration, while retaining explicit interpolation fallback.

**Architecture:** Reuse `plan_joint_path_rrt`, `validate_joint_path_in_mujoco`, and `execute_interpolated_joint_path` from `non_llm_transition.py`. Add a focused evaluator helper that splits the sampled 7D target into a 6D arm goal and one gripper target, returns structured planner/execution diagnostics, and runs through the demo's existing health/collision/replay wrapper.

**Tech Stack:** Python, NumPy, MuJoCo, pytest, existing Sci-VLA transition utilities.

## Constraints

- RRT plans only the six UR5e arm joints.
- The sampled seventh value restores the gripper after arm execution.
- RRT exhaustion must log `FALLBACK_RRT_TO_INTERPOLATION` and execute the direct path returned by the existing planner.
- Dataset trajectory/frame sampling and RRT use the same evaluation-seeded RNG instance.
- Every arm/gripper execution step retains warning detection, collision counting, and replay capture.
- Ordinary evaluation and retry restoration remain unchanged.

## Task 1: RED tests for RRT restore

**Files:**
- Modify: `scripts/autobio_scripts/test_problem_validation_demo.py`

1. Add a unit test for `restore_robot_state_rrt` proving current/target 6D values and the exact RNG reach `plan_joint_path_rrt`, returned RRT waypoints reach `execute_interpolated_joint_path`, and the seventh target restores the gripper afterward.
2. Add a fallback test proving status `FALLBACK_RRT_TO_INTERPOLATION`, direct waypoints, explicit log output, and successful gripper restoration.
3. Update evaluator integration tests to replace direct-restore expectations with RRT helper results, including planner diagnostics, collision/replay counts, and warning interruption.
4. Rename config/test usage from `interpolation_steps` to `restore_steps_per_segment`.
5. Run focused tests and observe failures caused by missing RRT helper/old direct wiring.

## Task 2: GREEN implementation

**Files:**
- Modify: `scripts/autobio_scripts/problem_validation_demo.py`
- Modify: `scripts/autobio_scripts/evaluator.py`

1. Rename `ProblemValidationDemoConfig.interpolation_steps` to `restore_steps_per_segment` and pass it through `execute_problem_validation_sequence`.
2. Add `restore_robot_state_rrt(...) -> dict` in `evaluator.py`:
   - validate finite target/current dimensions;
   - split arm/gripper indices;
   - plan with existing RRT and MuJoCo path validator;
   - execute returned waypoints with the configured steps per segment;
   - hold the target gripper control for a fixed 50 simulation steps;
   - return planner status/validation, waypoints, fallback flag, arm steps, and gripper steps.
3. Change only the fixed demo restore callback to call the RRT helper with `transition_rng`; retain its existing wrapped `task.step_and_log` health/collision/replay behavior.
4. Print planner status, validation, waypoint count, arm/gripper step counts, and fallback state.
5. Keep `restore_robot_state_direct` for retry restoration and existing callers.

## Task 3: Documentation and verification

**Files:**
- Modify: `README.md`

1. Update the demo description to state collision-aware 6D RRT, post-plan gripper restoration, and explicit direct-interpolation fallback.
2. Run:

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

git diff --check 534eb09..HEAD
```

3. Review the final diff for unintended changes to user-owned dirty files.
4. A real MP4 smoke run remains pending until the GPU policy endpoint is available.
