# LABVLA Integration Design

Date: 2026-06-29

## Goal

Add LABVLA as a comparison model path in Sci-VLA while keeping the existing OpenPI path intact. The integration should let the project generate training data once, fine-tune LABVLA on the same Sci-VLA demonstrations, and run simulation evaluation through the existing long-horizon evaluator.

The default simulated robot is a UR5e single-arm setup with a Robotiq 2F85-style gripper. Current task code exports 6 arm joint dimensions plus 1 gripper dimension, so the LABVLA contract for current Sci-VLA tasks is 7-dimensional state and action.

## Chosen Approach

Use a Sci-VLA-specific LABVLA schema and a thin inference adapter.

The existing LeRobot conversion should continue to write the current OpenPI-friendly feature names:

- `state`
- `actions`
- `observation/image`
- `observation/wrist_image`
- `observation/wrist_image_2`

LABVLA will learn this layout through a new schema instead of requiring a separate renamed dataset copy. Evaluation will adapt observation keys before sending requests to a LABVLA websocket server. LABVLA's deployment server already uses the OpenPI websocket protocol, so the evaluator can keep using `openpi_client.WebsocketClientPolicy`.

## Dataset Schema

Add a schema module under `third_party/labvla/schemas/scivla_ur5e_single_arm.py`.

The schema should use LABVLA's existing `SingleArmBlueprint`:

- `schema_id`: `scivla_ur5e_single_arm_v1`
- `robot_type`: `ur5e`
- arm degrees of freedom: `6`
- joint state/action columns: `state` and `actions`
- gripper state/action columns: `state` and `actions`
- arm action mode: `delta`
- gripper action mode: `abs`
- gripper semantic: `position`

Derived contract:

- `state_keys=("state",)`
- `state_dims=(7,)`
- `action_keys=("actions",)`
- `action_dims=(7,)`
- `delta_mask=(True, True, True, True, True, True, False)`
- `gripper_action_dims=(6,)`

Camera mapping should consume the existing Sci-VLA converted dataset columns:

- `observation/image` -> `image0`
- `observation/wrist_image` -> `image1`
- `observation/wrist_image_2` -> `image2`

This makes `python -m data_process stats --dataset <dataset> --schema scivla_ur5e_single_arm` operate on the same dataset produced by `scripts/convert.py`.

## Inference Adapter

Update `scripts/autobio_scripts/evaluate.py` to support a policy backend option, for example:

- `--policy-backend openpi` as the default
- `--policy-backend labvla` for LABVLA deployment

The OpenPI path should preserve current behavior.

The LABVLA path should transform each evaluator observation before calling websocket inference:

- copy `observation/state` to `state`
- keep `observation/state`
- map `observation/image` to `camera_1_rgb`
- map `observation/wrist_image` to `camera_2_rgb`
- map `observation/wrist_image_2` to `camera_3_rgb`
- keep `prompt`

The adapter should not alter action outputs. Both OpenPI and LABVLA servers return `{"actions": ...}` over the websocket protocol, and the evaluator should continue to execute that action chunk against `task_info["action_indices"]`.

The adapter should fail loudly if required LABVLA fields are missing. Missing state or all cameras should be treated as configuration errors, not silently replaced with zeros.

## Training And Deployment Flow

The existing data-generation commands remain valid:

```bash
python scripts/autobio_scripts/centrifuge5910_tasks.py
python scripts/autobio_scripts/thermal_cycler_tasks.py
bash scripts/autobio_scripts/render_all.bash logs/centrifuge5910_tasks
bash scripts/autobio_scripts/render_all.bash logs/thermal_cycler_tasks
python scripts/convert.py --data-dir logs/centrifuge5910_tasks --repo-id mani_centrifuge5910
python scripts/convert.py --data-dir logs/thermal_cycler_tasks --repo-id mani_thermalcycler
```

The README should add LABVLA-specific steps:

```bash
cd third_party/labvla
python -m data_process stats \
  --dataset ~/.cache/huggingface/lerobot/mani_centrifuge5910 \
  --schema scivla_ur5e_single_arm
```

Fine-tuning should use LABVLA's existing `scripts/train.py` and launcher patterns, with:

- `DataRoot` pointing at the LeRobot cache root containing the Sci-VLA repo id.
- `RepoIds` including the target Sci-VLA repo id, such as `mani_centrifuge5910`.
- `DatasetSchema=scivla_ur5e_single_arm`.
- `ExternalStatsPath` pointing at the stats file generated for that dataset, unless the selected LABVLA launcher auto-discovers the same file from the checkpoint or dataset metadata.

Deployment should use LABVLA's `deployment/serve_labvla.py` or `deployment/deploy.sh`, then evaluate from Sci-VLA with:

```bash
python scripts/autobio_scripts/evaluate.py \
  --policy-backend labvla \
  --host 127.0.0.1 \
  --port 8000 \
  --task centrifuge5910_long_task_1 \
  --time_limit 30 \
  --prompts "..."
```

## Error Handling

The schema should be importable by LABVLA's existing schema auto-loader.

The LABVLA policy adapter should raise clear `ValueError` messages when:

- `observation/state` is missing.
- A required camera key for the selected backend is absent.
- The returned action chunk is not 2-D or has a width other than `len(task_info["action_indices"])`.

The existing evaluator assertion already checks action width. The new adapter should focus on request-shape failures before the websocket call.

## Testing

Add focused tests that do not require GPUs or MuJoCo rendering:

- Schema import test:
  - import `third_party/labvla/schemas/scivla_ur5e_single_arm.py`
  - verify state/action dimensions are 7
  - verify `delta_mask` has six `True` values followed by `False`
  - verify camera mapping contains the three existing Sci-VLA columns

- Policy adapter test:
  - create a fake observation with state, three images, and prompt
  - verify `labvla` backend produces `state`, `camera_1_rgb`, `camera_2_rgb`, `camera_3_rgb`, and `prompt`
  - verify `openpi` backend leaves the observation unchanged
  - verify missing state or missing camera raises a clear error

- CLI compatibility test:
  - verify `parse_args()` accepts `--policy-backend labvla`
  - verify default remains `openpi`

## Non-Goals

This design does not rewrite `scripts/convert.py` to produce LABVLA-native camera column names.

This design does not modify simulation task definitions, action execution, long-horizon transition generation, or OpenPI training configs.

This design does not train or download LABVLA checkpoints as part of implementation.

## Acceptance Criteria

- Existing OpenPI evaluation commands still work without adding any new flags.
- A Sci-VLA LeRobot dataset generated by current scripts can be used by LABVLA stats/training with `--schema scivla_ur5e_single_arm`.
- `evaluate.py --policy-backend labvla` sends LABVLA-compatible observations to a LABVLA websocket server.
- Tests cover schema shape and adapter key mapping without requiring a running model server.
- README documents the LABVLA comparison path alongside the existing OpenPI path.
