"""`meta/labvla_manifest.json` parser + validator."""

from __future__ import annotations

import json
from pathlib import Path

from .camera_mapping import expand_camera_mapping
from .dataset_schema import DatasetSchema
from .errors import SchemaValidationError


MANIFEST_NAME = "labvla_manifest.json"
SUPPORTED_VERSION = 1


def load_manifest(path: str | Path) -> DatasetSchema:
    """Parse and validate a labvla_manifest.json file into a DatasetSchema.

    Raises ValueError on malformed manifests, always citing the manifest path
    so pre-construction parse errors point at WHICH manifest was broken.
    """
    path = Path(path)
    try:
        return _load_manifest_inner(path)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"manifest {path}: {type(e).__name__}: {e}") from e


def _load_manifest_inner(path: Path) -> DatasetSchema:
    with open(path) as f:
        data = json.load(f)

    version = data.get("version")
    if version != SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported manifest version {version!r} at {path} "
            f"(expected {SUPPORTED_VERSION})"
        )

    schema_id = data.get("schema_id")
    if not schema_id or not isinstance(schema_id, str):
        raise ValueError(f"manifest at {path} missing non-empty schema_id")

    robot_type = data.get("robot_type", "unknown")

    state = data.get("state") or {}
    action = data.get("action") or {}
    images = data.get("images") or {}
    _aec_raw = data.get("allow_extra_cameras", False)
    if not isinstance(_aec_raw, bool):
        raise ValueError(
            f"manifest allow_extra_cameras must be a JSON boolean, got "
            f"{_aec_raw!r} (M32: bool('false') is True — truthiness would "
            f"invert the camera cross-check)"
        )
    allow_extra_cameras = _aec_raw

    arm_layout = None
    if data.get("arm_layout") is not None:
        from .arm_layout import ArmLayoutSpec
        arm_layout = ArmLayoutSpec.from_dict(data["arm_layout"])

    # Optional auxiliary losses on annotation columns. Missing / empty list →
    # empty tuple → schema behaves exactly as before.
    annotation_losses: tuple = ()
    if data.get("annotation_losses"):
        from .annotation_loss import AnnotationLossSpec
        annotation_losses = tuple(
            AnnotationLossSpec.from_dict(d) for d in data["annotation_losses"]
        )

    state_keys = tuple(state.get("keys") or ())
    state_dims = tuple(state.get("dims") or ())
    action_keys = tuple(action.get("keys") or ())
    action_dims = tuple(action.get("dims") or ())
    _delta_raw = action.get("delta") or ()
    if not all(isinstance(v, bool) for v in _delta_raw):
        raise ValueError(
            f"manifest action.delta must be JSON booleans, got {_delta_raw!r} "
            f"(M32: truthiness would silently rewrite the delta mask)"
        )
    delta_mask = tuple(_delta_raw)
    gripper_action_dims = tuple(action.get("gripper_dims") or ())
    # Optional gripper physical semantic ("width" / "open_fraction" /
    # "position" / ...). Drives gripper canonicalization + cross-repo
    # compatibility guards in dataset_helpers. Absent in legacy manifests
    # → None → registry-based fallback keyed off schema_id still applies.
    gripper_semantic_raw = data.get("gripper_semantic")
    if gripper_semantic_raw is not None and not isinstance(gripper_semantic_raw, str):
        raise ValueError(
            f"manifest gripper_semantic must be a string or absent, got "
            f"{type(gripper_semantic_raw).__name__} (M32)"
        )
    gripper_semantic = gripper_semantic_raw or None
    # Virtual columns: parse this field explicitly so a manifest that declares
    # it does not silently drop it and then fail validation with
    # "virtual. key has no mapping".
    _vss_raw = data.get("virtual_state_sources") or {}
    if not isinstance(_vss_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in _vss_raw.items()
    ):
        raise ValueError(
            f"manifest virtual_state_sources must be a str->str map, got "
            f"{_vss_raw!r} (M32: no truthiness coercion at the wire boundary)"
        )
    virtual_state_sources = dict(_vss_raw)
    source_state = data.get("source_state") or {}
    source_action = data.get("source_action") or {}
    source_state_keys = tuple(source_state.get("keys") or ())
    source_state_dims = tuple(source_state.get("dims") or ())
    source_action_keys = tuple(source_action.get("keys") or ())
    source_action_dims = tuple(source_action.get("dims") or ())

    # Expand short image aliases ("image0") to the full unified key.
    expanded_images = expand_camera_mapping(images)

    # DatasetSchema's __post_init__ runs schema/validate.py::validate_schema,
    # which enforces every structural invariant. Wrap the construction so the
    # user sees the manifest path in the error message (instead of a terse
    # constructor trace) — otherwise authoring mistakes are hard to locate.
    try:
        return DatasetSchema(
            schema_id=schema_id,
            robot_type=robot_type,
            state_keys=state_keys,
            action_keys=action_keys,
            state_dims=state_dims,
            action_dims=action_dims,
            delta_mask=delta_mask,
            gripper_action_dims=gripper_action_dims,
            gripper_semantic=gripper_semantic,
            virtual_state_sources=virtual_state_sources,
            image_mapping=expanded_images,
            allow_extra_cameras=allow_extra_cameras,
            arm_layout=arm_layout,
            annotation_losses=annotation_losses,
            source_state_keys=source_state_keys,
            source_state_dims=source_state_dims,
            source_action_keys=source_action_keys,
            source_action_dims=source_action_dims,
            source="manifest",
            source_path=str(path),
        )
    except SchemaValidationError as e:
        raise SchemaValidationError(f"manifest {path}: {e}") from e
