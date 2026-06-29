# LABVLA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LABVLA as a comparison model path for Sci-VLA UR5e single-arm datasets and evaluation.

**Architecture:** Keep existing OpenPI behavior as the default. Add one LABVLA schema that describes the existing Sci-VLA LeRobot columns, and add a small observation adapter in `evaluate.py` that maps Sci-VLA observation keys to LABVLA websocket keys when `--policy-backend labvla` is selected.

**Tech Stack:** Python, pytest, LABVLA `DatasetSchema`/`SingleArmBlueprint`, existing OpenPI-compatible websocket policy protocol.

---

## File Structure

- Create `third_party/labvla/schemas/scivla_ur5e_single_arm.py`: LABVLA schema registration for Sci-VLA's UR5e + 2F85 7D state/action layout.
- Modify `scripts/autobio_scripts/evaluate.py`: add backend enum, observation adapter, and pass selected backend into serial and parallel policy creation.
- Create `tests/test_labvla_integration.py`: unit tests for schema shape, adapter behavior, and CLI default/override.
- Modify `README.md`: document LABVLA stats, fine-tuning, deployment, and evaluation commands.

## Task 1: Add Failing Schema Test

**Files:**
- Create: `tests/test_labvla_integration.py`
- Later create: `third_party/labvla/schemas/scivla_ur5e_single_arm.py`

- [ ] **Step 1: Write the failing schema test**

Create `tests/test_labvla_integration.py` with:

```python
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABVLA_ROOT = ROOT / "third_party" / "labvla"


def load_module(module_name: str, path: Path):
    if str(LABVLA_ROOT) not in sys.path:
        sys.path.insert(0, str(LABVLA_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scivla_ur5e_schema_matches_current_dataset_layout():
    module = load_module(
        "scivla_ur5e_single_arm",
        LABVLA_ROOT / "schemas" / "scivla_ur5e_single_arm.py",
    )

    schema = module.SCHEMA

    assert schema.schema_id == "scivla_ur5e_single_arm_v1"
    assert schema.robot_type == "ur5e"
    assert schema.state_keys == ("state",)
    assert schema.state_dims == (7,)
    assert schema.action_keys == ("actions",)
    assert schema.action_dims == (7,)
    assert schema.delta_mask == (True, True, True, True, True, True, False)
    assert schema.gripper_action_dims == (6,)
    assert dict(schema.image_mapping) == {
        "observation/image": "observation.images.image0",
        "observation/wrist_image": "observation.images.image1",
        "observation/wrist_image_2": "observation.images.image2",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_labvla_integration.py::test_scivla_ur5e_schema_matches_current_dataset_layout -v`

Expected: FAIL with `FileNotFoundError` for `third_party/labvla/schemas/scivla_ur5e_single_arm.py`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_labvla_integration.py
git commit -m "test: cover scivla labvla schema"
```

## Task 2: Implement LABVLA Schema

**Files:**
- Create: `third_party/labvla/schemas/scivla_ur5e_single_arm.py`
- Test: `tests/test_labvla_integration.py`

- [ ] **Step 1: Add the schema implementation**

Create `third_party/labvla/schemas/scivla_ur5e_single_arm.py`:

```python
"""Sci-VLA UR5e single-arm schema for LABVLA.

The current Sci-VLA conversion writes 6 UR5e joint dimensions plus one
Robotiq 2F85 gripper dimension into the same `state` / `actions` columns.
"""

from src.schema.blueprint import SingleArm, SingleArmBlueprint


SCHEMA = SingleArmBlueprint(
    schema_id="scivla_ur5e_single_arm_v1",
    robot_type="ur5e",
    arm=SingleArm(
        dof=6,
        joint_state_col="state",
        joint_action_col="actions",
        gripper_state_col="state",
        gripper_action_col="actions",
        arm_mode="delta",
        gripper_mode="abs",
    ),
    cameras={
        "observation/image": "image0",
        "observation/wrist_image": "image1",
        "observation/wrist_image_2": "image2",
    },
    source_path=__file__,
    gripper_semantic="position",
).build()
```

- [ ] **Step 2: Run the schema test to verify it passes**

Run: `pytest tests/test_labvla_integration.py::test_scivla_ur5e_schema_matches_current_dataset_layout -v`

Expected: PASS.

- [ ] **Step 3: Run a LABVLA registry import smoke test**

Run: `PYTHONPATH=third_party/labvla python -c "import schemas.scivla_ur5e_single_arm as s; print(s.SCHEMA.schema_id, s.SCHEMA.action_dims)"`

Expected output contains: `scivla_ur5e_single_arm_v1 (7,)`.

- [ ] **Step 4: Commit the schema**

```bash
git add third_party/labvla/schemas/scivla_ur5e_single_arm.py tests/test_labvla_integration.py
git commit -m "feat: add scivla labvla schema"
```

## Task 3: Add Failing Policy Adapter Tests

**Files:**
- Modify: `tests/test_labvla_integration.py`
- Later modify: `scripts/autobio_scripts/evaluate.py`

- [ ] **Step 1: Add evaluate module loader and adapter tests**

Append to `tests/test_labvla_integration.py`:

```python
import numpy as np


def load_evaluate_module():
    return load_module(
        "autobio_evaluate",
        ROOT / "scripts" / "autobio_scripts" / "evaluate.py",
    )


def make_fake_observation():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return {
        "observation/state": np.arange(7, dtype=np.float32),
        "observation/image": image,
        "observation/wrist_image": image + 1,
        "observation/wrist_image_2": image + 2,
        "prompt": "open the lid",
    }


def test_prepare_policy_observation_leaves_openpi_observation_unchanged():
    evaluate = load_evaluate_module()
    obs = make_fake_observation()

    prepared = evaluate.prepare_policy_observation(obs, policy_backend="openpi")

    assert prepared is obs


def test_prepare_policy_observation_maps_scivla_keys_for_labvla():
    evaluate = load_evaluate_module()
    obs = make_fake_observation()

    prepared = evaluate.prepare_policy_observation(obs, policy_backend="labvla")

    assert prepared is not obs
    assert prepared["state"] is obs["observation/state"]
    assert prepared["observation/state"] is obs["observation/state"]
    assert prepared["camera_1_rgb"] is obs["observation/image"]
    assert prepared["camera_2_rgb"] is obs["observation/wrist_image"]
    assert prepared["camera_3_rgb"] is obs["observation/wrist_image_2"]
    assert prepared["prompt"] == "open the lid"


def test_prepare_policy_observation_rejects_missing_labvla_state():
    evaluate = load_evaluate_module()
    obs = make_fake_observation()
    obs.pop("observation/state")

    try:
        evaluate.prepare_policy_observation(obs, policy_backend="labvla")
    except ValueError as exc:
        assert "observation/state" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_prepare_policy_observation_rejects_missing_labvla_camera():
    evaluate = load_evaluate_module()
    obs = make_fake_observation()
    obs.pop("observation/wrist_image")

    try:
        evaluate.prepare_policy_observation(obs, policy_backend="labvla")
    except ValueError as exc:
        assert "observation/wrist_image" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run: `pytest tests/test_labvla_integration.py -k "prepare_policy_observation" -v`

Expected: FAIL with `AttributeError` because `prepare_policy_observation` does not exist.

- [ ] **Step 3: Commit the failing adapter tests**

```bash
git add tests/test_labvla_integration.py
git commit -m "test: cover policy observation backend mapping"
```

## Task 4: Implement Policy Backend Adapter

**Files:**
- Modify: `scripts/autobio_scripts/evaluate.py`
- Test: `tests/test_labvla_integration.py`

- [ ] **Step 1: Add backend constants and observation adapter near `make_policy`**

In `scripts/autobio_scripts/evaluate.py`, replace the existing `make_policy` block:

```python
def make_policy(host: str, port: int) -> "Policy":
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ws_policy = WebsocketClientPolicy(host, port)
    def policy_fn(obs: dict) -> np.ndarray:
        return ws_policy.infer(obs)['actions']
    return policy_fn
```

with:

```python
POLICY_BACKENDS = ("openpi", "labvla")


def prepare_policy_observation(obs: dict, policy_backend: str) -> dict:
    if policy_backend == "openpi":
        return obs

    if policy_backend != "labvla":
        raise ValueError(f"Unsupported policy backend: {policy_backend}")

    required_keys = [
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "observation/wrist_image_2",
    ]
    missing = [key for key in required_keys if key not in obs or obs[key] is None]
    if missing:
        raise ValueError(
            "LABVLA policy backend requires observation keys "
            f"{required_keys}; missing {missing}"
        )

    return {
        **obs,
        "state": obs["observation/state"],
        "camera_1_rgb": obs["observation/image"],
        "camera_2_rgb": obs["observation/wrist_image"],
        "camera_3_rgb": obs["observation/wrist_image_2"],
    }


def make_policy(host: str, port: int, policy_backend: str = "openpi") -> "Policy":
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ws_policy = WebsocketClientPolicy(host, port)

    def policy_fn(obs: dict) -> np.ndarray:
        request_obs = prepare_policy_observation(obs, policy_backend)
        return ws_policy.infer(request_obs)["actions"]

    return policy_fn
```

- [ ] **Step 2: Update worker globals and initializer signature**

In `scripts/autobio_scripts/evaluate.py`, add:

```python
_policy_backend: str
```

after the existing global declarations.

Change `init_worker(...)` to accept `policy_backend: str` after `port: int`, add `_policy_backend` to the `global` statement, and change:

```python
_policy = make_policy(host, port)
```

to:

```python
_policy_backend = policy_backend
_policy = make_policy(host, port, policy_backend)
```

- [ ] **Step 3: Add the CLI argument**

In `parse_args()`, after the `--port` argument, add:

```python
    parser.add_argument(
        "--policy-backend",
        type=str,
        default="openpi",
        choices=POLICY_BACKENDS,
        help="Policy websocket payload adapter to use: openpi keeps current keys; labvla maps Sci-VLA keys to LABVLA camera/state keys.",
    )
```

- [ ] **Step 4: Pass backend through serial and parallel paths**

In the serial path, change:

```python
policy = make_policy(args.host, args.port)
```

to:

```python
policy = make_policy(args.host, args.port, args.policy_backend)
```

In the `ProcessPoolExecutor` `initargs`, insert `args.policy_backend` immediately after `args.port`, matching the updated `init_worker` signature.

- [ ] **Step 5: Run adapter tests to verify they pass**

Run: `pytest tests/test_labvla_integration.py -k "prepare_policy_observation" -v`

Expected: PASS.

- [ ] **Step 6: Commit the adapter**

```bash
git add scripts/autobio_scripts/evaluate.py tests/test_labvla_integration.py
git commit -m "feat: add labvla policy backend adapter"
```

## Task 5: Add Failing CLI Tests

**Files:**
- Modify: `tests/test_labvla_integration.py`
- Modify: `scripts/autobio_scripts/evaluate.py`

- [ ] **Step 1: Add CLI tests**

Append to `tests/test_labvla_integration.py`:

```python
def test_parse_args_defaults_to_openpi_backend(monkeypatch):
    evaluate = load_evaluate_module()
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    args = evaluate.parse_args()

    assert args.policy_backend == "openpi"


def test_parse_args_accepts_labvla_backend(monkeypatch):
    evaluate = load_evaluate_module()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--policy-backend", "labvla"])

    args = evaluate.parse_args()

    assert args.policy_backend == "labvla"
```

- [ ] **Step 2: Run CLI tests**

Run: `pytest tests/test_labvla_integration.py -k "parse_args" -v`

Expected: PASS if Task 4 already added the argument. If it fails, fix only the `parse_args()` argument wiring shown in Task 4 Step 3.

- [ ] **Step 3: Commit CLI tests**

```bash
git add tests/test_labvla_integration.py scripts/autobio_scripts/evaluate.py
git commit -m "test: cover policy backend cli"
```

## Task 6: Document LABVLA Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add LABVLA installation note**

Under `## Installation (editable local install)`, after the OpenPI install block, add:

```markdown
For LABVLA comparison experiments, create the LABVLA environment separately:

```bash
conda create -n labvla python=3.10 -y
conda activate labvla
cd third_party/labvla
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install flash_attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
```
```

- [ ] **Step 2: Add LABVLA stats and fine-tuning section**

Under `## Fine-tuning`, after the OpenPI commands, add:

```markdown
### LABVLA fine-tuning

Sci-VLA UR5e demonstrations use the LABVLA schema `scivla_ur5e_single_arm`.
The existing `scripts/convert.py` output can be reused directly.

```bash
cd third_party/labvla
python -m data_process stats \
  --dataset ~/.cache/huggingface/lerobot/mani_centrifuge5910 \
  --schema scivla_ur5e_single_arm
```

Use LABVLA's training launcher or `scripts/train.py` with the Sci-VLA repo id,
the LeRobot cache root, and `DatasetSchema=scivla_ur5e_single_arm`. For example,
set the launcher variables to:

```bash
DataRoot=~/.cache/huggingface/lerobot
RepoIds=mani_centrifuge5910
DatasetSchema=scivla_ur5e_single_arm
ExternalStatsPath=~/.cache/huggingface/lerobot/mani_centrifuge5910/meta/stats.json
```
```

- [ ] **Step 3: Add LABVLA evaluation section**

Under `## Evaluation`, after the OpenPI websocket serve/evaluate example, add:

```markdown
### Evaluate LABVLA on simulations

Start a LABVLA websocket server from the LABVLA environment:

```bash
cd third_party/labvla
python deployment/serve_labvla.py \
  --pretrained_path /path/to/labvla/checkpoint \
  --port 8000 \
  --device cuda \
  --training_repo_id mani_centrifuge5910
```

Then run Sci-VLA evaluation with the LABVLA observation adapter:

```bash
python ./scripts/autobio_scripts/evaluate.py \
  --policy-backend labvla \
  --host 127.0.0.1 \
  --port 8000 \
  --task 'centrifuge5910_long_task_1' \
  --time_limit 30 \
  --prompts "open the lid of the centrifuge5910,pick the experimental centrifuge tube from rack and place it into the centrifuge5910,pick the balance centrifuge tube from rack and place it into the centrifuge5910,close the lid of the centrifuge5910,press the screen button to start the centrifuge5910"
```
```

- [ ] **Step 4: Run README grep checks**

Run: `rg -n "scivla_ur5e_single_arm|policy-backend labvla|serve_labvla" README.md`

Expected: output includes all three terms.

- [ ] **Step 5: Commit docs**

```bash
git add README.md
git commit -m "docs: add labvla comparison workflow"
```

## Task 7: Final Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_labvla_integration.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Run existing lightweight tests**

Run: `pytest tests/test_split_libero_long_subtasks.py tests/test_generate_libero_recovery_subtasks.py -v`

Expected: all tests PASS.

- [ ] **Step 3: Run LABVLA schema smoke import**

Run: `PYTHONPATH=third_party/labvla python -c "import schemas.scivla_ur5e_single_arm as s; assert s.SCHEMA.action_dims == (7,); print('ok')"`

Expected output: `ok`.

- [ ] **Step 4: Inspect git status**

Run: `git status --short`

Expected: no uncommitted changes except any intentionally untracked upstream `third_party/labvla/` files that existed before implementation.

- [ ] **Step 5: Final commit if verification required any fixups**

If Step 1, 2, or 3 required fixups, commit them:

```bash
git add scripts/autobio_scripts/evaluate.py third_party/labvla/schemas/scivla_ur5e_single_arm.py tests/test_labvla_integration.py README.md
git commit -m "fix: verify labvla integration"
```

If no fixups were required, do not create an empty commit.

## Self-Review

Spec coverage:
- LABVLA schema is covered by Tasks 1 and 2.
- Inference adapter and default OpenPI behavior are covered by Tasks 3, 4, and 5.
- README workflow is covered by Task 6.
- Verification without GPU or MuJoCo rendering is covered by Task 7.

Placeholder scan:
- No unresolved placeholder markers or deferred implementation language remains.
- Every code-changing step includes concrete code or exact replacement guidance.

Type consistency:
- The adapter function is consistently named `prepare_policy_observation`.
- CLI flag is consistently `--policy-backend`.
- Schema file and schema name are consistently `scivla_ur5e_single_arm`.
