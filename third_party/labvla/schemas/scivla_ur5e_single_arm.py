"""Sci-VLA UR5e single-arm schema for LABVLA.

The current Sci-VLA conversion writes 6 UR5e joint dimensions plus one
Robotiq 2F85 gripper dimension into the same `state` / `actions` columns.
"""

from dataclasses import replace

from src.schema.blueprint import SingleArm, SingleArmBlueprint


_SCHEMA = SingleArmBlueprint(
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

# SingleArmBlueprint normalizes source camera names. Sci-VLA's converted
# LeRobot columns intentionally contain slash-style keys, so restore them after
# build while keeping the blueprint-derived state/action layout.
SCHEMA = replace(
    _SCHEMA,
    image_mapping={
        "observation/image": "observation.images.image0",
        "observation/wrist_image": "observation.images.image1",
        "observation/wrist_image_2": "observation.images.image2",
    },
)

del _SCHEMA
