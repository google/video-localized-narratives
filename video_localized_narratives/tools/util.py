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

"""Utilities related to Video Localized Narratives."""

import glob
import json
import os
from typing import Any

import numpy as np

from pathlib import Path

from video_localized_narratives.tools import frame

JsonData = dict[str, Any]


def frame_number_from_filename(filename: str) -> int:
  stem = Path(filename).stem
  if stem[:4] == 'img_':
    stem = stem[4:]
  return int(stem)


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
  color = np.array(overlay_color)
  overlaid = (alpha * color + (1.0 - alpha) * img).astype(img.dtype)
  return np.where(mask[..., np.newaxis], overlaid, img)


def load_json_data(filename: str) -> JsonData:
  with open(filename) as f:
    return json.load(f)


def get_all_frames(folder: str) -> list[frame.Frame]:
  frames = glob.glob(os.path.join(folder, '*.jpg'))
  if frames:
    jpg = True
  else:
    frames = glob.glob(os.path.join(folder, '*.png'))
    jpg = False

  names = sorted(frames, key=frame_number_from_filename)
  stems = [Path(n).stem for n in names]
  return [frame.Frame(s, folder, jpg) for s in stems]
