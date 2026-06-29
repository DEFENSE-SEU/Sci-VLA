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
