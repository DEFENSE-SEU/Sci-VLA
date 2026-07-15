# Problem Validation Demo Design

## Goal

Add a one-command evaluation demo that records a continuous video of a fixed two-task VLA workflow:

1. Infer `open the lid of the thermal cycler` until the local success predicate first passes.
2. Continue VLA inference and action execution for five additional seconds of simulation time.
3. Sample one trajectory for `place pcrPlate into the thermal cycler`, then randomly sample one robot state from the first 30% of that trajectory.
4. Directly interpolate the robot arm and gripper to the sampled state.
5. Resume VLA inference for `place pcrPlate into the thermal cycler` until success or timeout.
6. Save the entire uninterrupted rollout as video.

## User Interface

`scripts/autobio_scripts/evaluate.py` gains a Boolean `--problem-validation-demo` option. When enabled, it applies the complete fixed profile:

- task: `thermal_cycler_long_task_1`
- prompts, in order:
  - `open the lid of the thermal cycler`
  - `place pcrPlate into the thermal cycler`
- one episode
- serial evaluation
- replay video enabled
- local task predicates used for phase completion
- no ordinary transition generation between the two prompts

Host, port, policy backend, control frequency, per-prompt time limit, video FPS, render device, MuJoCo backend, log path, and master seed continue to use the existing CLI options. The local LeRobot repository is `mani_thermalcycler`.

The demo profile takes precedence over conflicting generic task, prompt, worker-count, episode-count, video, intervention, experiment-mode, and transition-mode values. The CLI prints the effective fixed profile so this override is visible.

## Architecture

### Demo profile

A small immutable configuration in `evaluate.py` defines the fixed task, prompts, five-second post-success duration, dataset repository, 30% prefix fraction, and interpolation length. A resolver applies it to the parsed arguments before task or worker creation. Keeping the profile in one place prevents the fixed workflow from being reconstructed inconsistently across serial and worker paths.

The fixed demo runs only through the serial evaluator path. This guarantees that one policy connection, one simulator instance, and one replay buffer own the full video.

### Dataset state sampler

A focused helper module reads the local `mani_thermalcycler` LeRobot metadata and trajectory data. It:

1. Matches the exact normalized task text `place pcrPlate into the thermal cycler`.
2. Builds the candidate episode list containing that task.
3. Uses the episode RNG seeded from the evaluation seed to select one candidate episode.
4. Defines the prefix as frames `0` through `ceil(0.30 * episode_length) - 1`, inclusive.
5. Uses the same RNG to select one frame uniformly from that non-empty prefix.
6. Reads only the seven-value `state` field for the selected frame.

The helper returns both the state and structured provenance: repository, task, episode index, episode length, prefix frame count, frame index, and normalized frame ratio. Sampling the episode and frame with the evaluation RNG makes the demo reproducible for a given seed.

### Demo phase control

`Evaluator.evaluate` accepts an optional demo configuration. Normal evaluations retain the current behavior.

The fixed demo follows these phases:

1. `FIRST_PROMPT`: run the first prompt and check its local success predicate after every executed policy action.
2. `POST_SUCCESS_TAIL`: latch the first successful simulation timestamp. Continue requesting and executing VLA actions with the first prompt until at least five further seconds of simulation time have elapsed. Later predicate changes do not cancel the latched success.
3. `SAMPLE_TARGET`: sample the second-task dataset state and log its provenance.
4. `INTERPOLATE`: linearly interpolate all seven action controls—six arm joints and the gripper—from the current robot state to the sampled state. Each simulation step uses the existing task logging and replay capture path.
5. `SECOND_PROMPT`: switch the observation prompt and run normal non-timeout VLA inference until the second local predicate succeeds or the prompt time limit expires.
6. `FINALIZE`: finish the task, finalize render extras, and save the replay video.

The five-second tail belongs only to the first prompt. The first prompt's normal time limit governs how long the evaluator waits for initial success; the tail begins only after success and is not charged against that limit.

## State Restoration

The LeRobot `state` schema for this dataset is seven-dimensional: six UR5e joint positions followed by the gripper state. Restoration interpolates all seven components together through the task's seven action indices. It does not reset object, instrument, clock, or other MuJoCo state, so the open thermal-cycler lid produced by the first task remains in the scene for the placement task.

The interpolation is intentionally direct and does not invoke retrieval ranking, collision planning, RRT, or an LLM transition agent. This is the behavior under demonstration.

## Video and Diagnostics

The existing replay buffer remains active from evaluator reset through both prompts and interpolation. The demo writes a filename beginning with `problem_validation_open_lid_place_pcr_plate`. It always writes the front camera MP4 and also writes the left camera MP4 when that camera exists.

Diagnostics identify:

- the first-success simulation timestamp;
- the end of the five-second tail;
- sampled episode and frame provenance;
- the seven-dimensional target state;
- interpolation step count;
- first-prompt and second-prompt success;
- final video path through the existing video writer.

## Failure Handling

- Missing local dataset or metadata: fail with the expected repository name and cache location guidance.
- Missing second-task episodes: fail with the exact required task text.
- Empty trajectory or prefix: fail with the selected episode index and length.
- Non-finite or non-seven-dimensional state: reject it before interpolation.
- Unhealthy MuJoCo simulation: stop the current phase and mark the demo failed.
- First-prompt timeout: do not sample or interpolate; finalize a failure video.
- Sampling or interpolation error after simulation starts: finalize a failure video, then report the original error.
- Second-prompt timeout: finalize a failure video and report second-task failure.

All finalization is centralized so render cleanup and video writing occur once.

## Testing

Unit and focused evaluator tests cover:

- the demo CLI/profile overriding generic conflicting options;
- exact task filtering in the dataset sampler;
- deterministic episode and frame selection for a fixed seed;
- sampled frame membership in `[0, ceil(0.30 * length) - 1]`;
- short trajectories still producing a non-empty one-frame prefix;
- rejection of wrong-sized and non-finite states;
- first success being latched;
- five additional simulation seconds of VLA calls and action execution;
- direct seven-dimensional arm-and-gripper interpolation;
- no sampling or second prompt after first-task failure;
- second prompt starting only after interpolation completes;
- replay capture spanning all phases and finalization occurring once.

A final local verification runs the focused pytest files and the surrounding evaluate/transition tests. If a compatible VLA policy server is reachable, a one-episode smoke run produces the requested MP4; otherwise the implementation handoff includes the exact command and identifies the policy-server dependency.

## Scope

This change adds only the fixed thermal-cycler validation demonstration. It does not generalize trajectory-prefix restoration to arbitrary tasks, add video overlays, alter model serving, change dataset generation, or modify existing experiment modes.
