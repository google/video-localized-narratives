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

"""Class for a frame of a video."""

import dataclasses
import os
from typing import Optional

import numpy as np
import PIL.Image


@dataclasses.dataclass(frozen=True)
class Frame:
  """A frame of a video that can be loaded as an np.ndarray."""

  name: str
  root_folder: Optional[str]
  is_jpg: Optional[bool] = None

  def __str__(self):
    if self.root_folder is None:
      return self.name
    return os.path.join(self.root_folder, self.name)

  def load(self) -> np.ndarray:
    """Loads the frame as an np.ndarray. Works both for jpg and png."""
    assert self.root_folder is not None
    base = os.path.join(self.root_folder, self.name)

    if self.is_jpg is not None:
      if self.is_jpg:
        return load_img(base + '.jpg')
      else:
        return load_img(base + '.png')

    # First try jpg and if it is not found, try png instead.
    try:
      return load_img(base + '.jpg')
    except FileNotFoundError:
      return load_img(base + '.png')


@dataclasses.dataclass(frozen=True)
class KeyFrame(Frame):
  keyframe_idx: Optional[int] = None


def load_img(filename: str) -> np.ndarray:
  with open(filename, 'rb') as f:
    return np.array(PIL.Image.open(f))

