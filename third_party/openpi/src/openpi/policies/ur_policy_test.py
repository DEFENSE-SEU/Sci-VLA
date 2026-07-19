import numpy as np

from openpi.models import model as _model
from openpi.policies import ur_policy


def test_ur_inputs_for_pi05():
    base_image = np.full((3, 4, 5), 0.5, dtype=np.float32)
    wrist_image = np.full((3, 4, 5), 0.25, dtype=np.float32)
    joints = np.arange(6, dtype=np.float32)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    result = ur_policy.URInputs(model_type=_model.ModelType.PI05)(
        {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/joints": joints,
            "observation/gripper": np.float32(0.75),
            "actions": actions,
            "prompt": b"close lid",
        }
    )

    np.testing.assert_array_equal(result["state"], np.append(joints, 0.75))
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros((4, 5, 3), dtype=np.uint8))
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["prompt"] == "close lid"


def test_ur_outputs_strip_model_padding():
    actions = np.arange(96, dtype=np.float32).reshape(3, 32)

    result = ur_policy.UROutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :7])
