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

"""Provides the VideoLocalizedNarrativeDataset class for loading of VidLNs."""

import json
from typing import Optional

from video_localized_narratives.tools import vidln


class VideoLocalizedNarrativeDataset:
  """A dataset of videos with Video Localized Narratives."""

  def __init__(self, jsonl_filename: str, frames_path: Optional[str]):
    self._vidlns: list[vidln.VideoLocalizedNarrative] = []
    with open(jsonl_filename) as f:
      for l in f:
        raw_data = json.loads(l)
        vln = vidln.VideoLocalizedNarrative(raw_data, frames_path)
        self._vidlns.append(vln)

  def __getitem__(self, idx: int) -> vidln.VideoLocalizedNarrative:
    return self._vidlns[idx]

  def __len__(self) -> int:
    return len(self._vidlns)
