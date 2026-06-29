"""
LabUtopia <-> LeRobot data transforms.

Handles converting LabUtopia observation format (camera_1_rgb, camera_2_rgb, camera_3_rgb,
state, prompt) into the LeRobot model input format, and converting model action outputs
back into the format LabUtopia expects.

This module is analogous to openpi's labsim_policy.py (LabSimInputs / LabSimOutputs).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """
    Parse an image to uint8 (H, W, C) format.

    LabUtopia sends images as uint8 (H, W, C), but we handle edge cases:
    - float images in [0, 1] -> convert to uint8
    - (C, H, W) layout -> transpose to (H, W, C)
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    # If shape is (C, H, W) with C in {1, 3, 4}, transpose
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    return image


def image_hwc_uint8_to_chw_float32(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Convert (H, W, C) uint8 image to (C, H, W) float32 tensor.

    Args:
        image: uint8 image in (H, W, C) format
        normalize: if True, output [0, 1] range (matches training data from video_utils);
                   if False, output [0, 255] range (for models with own image_to_float
                   step in their preprocessor, e.g. XVLA has xvla_image_to_float)
    """
    image = np.asarray(image, dtype=np.float32)
    if normalize:
        image = image / 255.0  # -> [0, 1] to match training data format
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    return torch.from_numpy(image)


def resize_image_tensor(image_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize a (C, H, W) or (1, C, H, W) tensor to (target_h, target_w).
    Returns (C, H, W) tensor.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        image_tensor.float(), size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)


class LabUtopiaInputTransform:
    """
    Transform LabUtopia observation dict into a LeRobot preprocessor-ready batch dict.

    LabUtopia RemoteInferenceEngine sends:
        - camera_1_rgb: (H, W, C) uint8 ndarray
        - camera_2_rgb: (H, W, C) uint8 ndarray
        - camera_3_rgb: (H, W, C) uint8 ndarray  (arm camera, optional)
        - state: (8,) float array (7 joints + 1 gripper)
        - prompt / language_instruction: str

    The LeRobot preprocessor pipeline expects a flat dict with keys like:
        - observation.images.camera_1_rgb: (C, H, W) float tensor in [0, 255]
        - observation.images.camera_2_rgb: (C, H, W) float tensor in [0, 255]
        - observation.state: (state_dim,) float tensor
        - task: str

    The preprocessor's RenameObservationsProcessorStep will rename these keys to
    match the model's feature names (e.g., camera_1_rgb -> image, camera_2_rgb -> image2).
    Then AddBatchDimension, Tokenizer, ImageToFloat, ImageNetNormalize, Device, Normalizer
    steps handle the rest.
    """

    def __init__(
        self,
        image_features: Dict[str, Tuple[int, ...]],
        camera_to_obs_key: Dict[str, str],
        state_dim: int = 8,
        normalize_images: bool = True,
    ):
        """
        Args:
            image_features: dict mapping LeRobot feature names to (C, H, W) shapes.
                e.g. {"observation.images.image": (3, 256, 256), ...}
            camera_to_obs_key: dict mapping LabUtopia camera names to the ORIGINAL
                observation key (before rename), e.g.
                {"camera_1_rgb": "observation.images.camera_1_rgb", ...}
            state_dim: dimension of the state vector.
            normalize_images: if True, normalize images to [0, 1] (for most models);
                if False, keep [0, 255] (for models like XVLA whose preprocessor
                has its own image_to_float step that divides by 255).
        """
        self.image_features = image_features
        self.camera_to_obs_key = camera_to_obs_key
        self.state_dim = state_dim
        self.normalize_images = normalize_images
        # Derive a single target (H, W) from image_features when every visual
        # feature declares the same spatial size. If shapes are missing or
        # inconsistent, leave _target_hw = None and skip resizing rather than
        # guessing one ambiguously.
        self._target_hw: Optional[Tuple[int, int]] = None
        hw_set = set()
        for shape in (image_features or {}).values():
            if isinstance(shape, (tuple, list)) and len(shape) == 3:
                hw_set.add((int(shape[1]), int(shape[2])))
        if len(hw_set) == 1:
            self._target_hw = next(iter(hw_set))

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw LabUtopia observation to a LeRobot preprocessor input batch.

        Returns a dict with keys like "observation.images.camera_1_rgb",
        "observation.state", "task" — ready to be fed into the preprocessor pipeline.

        Images are resized to the target (H, W) from `image_features` (when
        uniform) and the state vector is padded/truncated to `state_dim`. Both
        are no-ops when the inputs already match the trained resolution / width.
        """
        batch = {}

        # ---- State (pad/truncate/validate to state_dim) ----
        state = np.asarray(obs.get("state", np.zeros(self.state_dim)), dtype=np.float32)
        state = np.atleast_1d(state).reshape(-1)
        if state.shape[0] != self.state_dim:
            logger.warning(
                "LabUtopiaInputTransform: state dim %d != expected state_dim %d; "
                "padding/truncating to match the trained contract.",
                state.shape[0], self.state_dim,
            )
            fitted = np.zeros(self.state_dim, dtype=np.float32)
            n = min(state.shape[0], self.state_dim)
            fitted[:n] = state[:n]
            state = fitted
        batch["observation.state"] = torch.from_numpy(state).float()

        # ---- Images (resize to image_features target H,W when known) ----
        for lab_cam_name, obs_key in self.camera_to_obs_key.items():
            if lab_cam_name in obs:
                img_hwc = parse_image_to_uint8_hwc(obs[lab_cam_name])
                # Convert to (C, H, W) float in [0, 255] range
                img_tensor = image_hwc_uint8_to_chw_float32(img_hwc, normalize=self.normalize_images)
                if self._target_hw is not None and tuple(img_tensor.shape[-2:]) != self._target_hw:
                    img_tensor = resize_image_tensor(
                        img_tensor, self._target_hw[0], self._target_hw[1]
                    )
                batch[obs_key] = img_tensor

        # ---- Language instruction / prompt ----
        prompt = obs.get("prompt", obs.get("language_instruction", ""))
        if isinstance(prompt, (np.ndarray, np.generic)):
            prompt = str(prompt.item()) if hasattr(prompt, 'item') else str(prompt)
        # Always write "task" to batch to avoid KeyError in TokenizerProcessorStep.
        # If prompt is empty, use an empty string as placeholder.
        batch["task"] = prompt if prompt else ""



        return batch


