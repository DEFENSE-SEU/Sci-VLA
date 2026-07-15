# Problem Validation Demo RRT Restore Design

## Goal

Replace the fixed demo's direct seven-dimensional restoration with collision-aware RRT planning to the sampled second-task robot state, while preserving the existing deterministic sampling, five-second post-success VLA tail, continuous replay, strict video delivery, and normal evaluation behavior.

If RRT cannot find a valid path, the demo must explicitly report `FALLBACK_RRT_TO_INTERPOLATION` and use direct arm-joint interpolation so the fixed demonstration can continue.

## Unchanged Workflow

The demo still:

1. Runs `open the lid of the thermal cycler` until local success.
2. Continues VLA inference and action execution for at least five additional simulation seconds.
3. Randomly selects a `place pcrPlate into the thermal cycler` trajectory and uniformly selects one state from its first 30% of frames.
4. Uses the evaluation seed for reproducibility.
5. Runs the placement prompt after restoration.
6. Captures one uninterrupted front video and, when available, one uninterrupted left video.

## RRT Restoration

The sampled LeRobot state remains seven-dimensional. Its first six values are the UR5e arm-joint target; its seventh value is the gripper target.

The restore phase will:

1. Read the current six arm joint positions from the task's state indices.
2. Plan from the current arm state to the sampled six-dimensional target with the existing `plan_joint_path_rrt` implementation.
3. Validate every RRT edge with `validate_joint_path_in_mujoco`, using the current simulator state and the UR5e joint span.
4. Use the existing transition RNG, after trajectory/frame sampling, for RRT sampling. A fixed evaluation seed therefore reproduces the complete sampling and planning sequence.
5. Execute the returned RRT waypoints by interpolating each waypoint edge at the configured control resolution. This is low-level execution of the RRT path, not a direct start-to-goal interpolation.
6. Restore the sampled gripper target after the arm reaches the target. The gripper is not part of the RRT search because the existing collision validator and planning space are defined for the six UR5e arm joints.

The planner result and execution result will be represented by a focused demo restore result containing planner status, validation payload, waypoint count, executed arm steps, gripper steps, collision count, and fallback state.

## Fallback

The existing RRT planner returns the direct start/goal path with status `FALLBACK_RRT_TO_INTERPOLATION` when no valid RRT route is found within its iteration budget.

The demo will preserve that behavior:

- print an explicit fallback diagnostic before execution;
- execute the direct start-to-goal arm interpolation returned by the planner;
- restore the target gripper state;
- continue to the second VLA prompt if simulation remains healthy.

Fallback is a recorded planner outcome, not an episode failure by itself.

## Simulation Health, Replay, and Metrics

Every executed arm or gripper simulation step must:

- call the task's original `step_and_log`;
- stop immediately on any MuJoCo warning;
- count robot-object contacts;
- capture replay frames without clearing the existing replay buffer.

The restore phase contributes one transition to existing timing and collision summaries. Diagnostics include sampled task/dataset provenance plus:

- `planner_status`;
- `planner_validation`;
- `waypoint_count`;
- `executed_arm_steps`;
- `executed_gripper_steps`;
- `fallback=true|false`.

Strict front/left MP4 write behavior remains unchanged: the fixed demo cannot report success when a required video write fails.

## Interfaces and Scope

The generic RRT implementation in `non_llm_transition.py` will be reused rather than duplicated. A focused restore helper in the problem-validation domain accepts the model, data, task, sampled target, RNG, replay/collision step callback, and execution settings.

The existing `restore_robot_state_direct` helper remains available for prompt-retry restoration and other callers. Only the fixed problem-validation demo switches to RRT.

The demo configuration's misleading direct-interpolation field will be renamed to an RRT execution resolution such as `restore_steps_per_segment`. Internal sequence callback names will be updated accordingly.

This change does not alter:

- ordinary single- or multi-prompt evaluation;
- existing baseline/ablation transition modes;
- dataset generation or sampling semantics;
- task success predicates;
- policy serving;
- video naming.

## Failure Handling

- Invalid sampled state: reject before planning.
- Planner/validator exception: finalize a failure video and preserve the original exception.
- MuJoCo warning during arm or gripper execution: stop immediately, skip the second prompt, finalize a failure video, and surface a contextual error.
- RRT exhaustion: use the explicit interpolation fallback rather than failing.
- Required MP4 write failure: retain the current strict demo failure behavior.

## Testing

Tests will verify:

- RRT receives the current and sampled six-dimensional arm states;
- the same transition RNG is passed to trajectory sampling and RRT planning;
- validated RRT waypoints, rather than a direct start/goal path, drive arm execution;
- `FALLBACK_RRT_TO_INTERPOLATION` is logged and executed as a direct path without failing the episode;
- the seventh sampled value restores the gripper after arm execution;
- arm and gripper execution both capture replay, count contacts, and stop on MuJoCo warnings;
- planner status, validation, waypoint/step counts, and fallback state appear in diagnostics/timing data;
- first-task failure still prevents sampling/planning;
- ordinary evaluate paths and direct retry restoration remain unchanged.

The focused regression suite must pass before a real GPU/policy-server smoke run. Actual MP4 generation remains dependent on a reachable policy endpoint and working GPU runtime.
