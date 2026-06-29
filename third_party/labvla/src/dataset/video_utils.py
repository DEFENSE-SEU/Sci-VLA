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
"""Video codec helper — live remainder after the 2026-06-11 dead-code sweep.

The torchcodec/torchvision decode stack that lived here served the deleted
lerobot dataset modules; training decodes video through
``src/adapters/lerobot_base._read_video_frame`` (shared PyAV container cache).
"""
from __future__ import annotations

import importlib
import logging


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    logging.warning(
        "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
    )
    return "pyav"
