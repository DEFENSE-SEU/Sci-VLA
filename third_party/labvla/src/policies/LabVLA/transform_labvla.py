"""Back-compat shim.

``Qwen3_VLProcessorTransformFn`` / ``UnifyLabVLAInputsTransformFn`` are DATA
TRANSFORMS and now live in ``transforms.vlm_processor``. Import-only re-export:
the classes are defined (and draccus-registered) exactly once, in their new
home. Existing checkpoints/configs and any out-of-tree caller importing this
path keep working.
"""
from src.transforms.vlm_processor import (  # noqa: F401
    Qwen3_VLProcessorTransformFn,
    UnifyLabVLAInputsTransformFn,
)

__all__ = ["Qwen3_VLProcessorTransformFn", "UnifyLabVLAInputsTransformFn"]
