#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remaining lerobot training utility.

The checkpoint layout helpers that used to live here are superseded by
scripts/utils/checkpoint.py (atomic training_state.pt + accelerate_state).
train.py's only live import from this module is ``update_last_checkpoint``.
"""
from pathlib import Path

from src.utils.constants import LAST_CHECKPOINT_LINK


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Update the 'last' symlink to point at the newly saved checkpoint.

    Atomic: create a tmp symlink then os.replace, so a crash can't leave the
    'last' pointer missing.
    """
    import os as _os
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    tmp_link = last_checkpoint_dir.with_name(last_checkpoint_dir.name + ".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(relative_target)
    _os.replace(str(tmp_link), str(last_checkpoint_dir))
    return last_checkpoint_dir
