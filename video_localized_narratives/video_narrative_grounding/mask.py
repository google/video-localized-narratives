# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides the Mask class for a segmentation mask in a single frame."""


from typing import Any
import numpy as np
from pycocotools import mask as cocomask


class Mask:
  """A segmentation mask in a single frame of a video."""

  def __init__(self, rle: dict[str, Any]):
    self._rle = rle

  def load(self) -> np.ndarray:
    return cocomask.decode(self._rle)
