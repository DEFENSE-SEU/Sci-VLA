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
