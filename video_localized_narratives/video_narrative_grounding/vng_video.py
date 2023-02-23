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

"""Provides the VNGVideo class for easy use of VNG data."""

from typing import Optional

from video_localized_narratives.tools import util
from video_localized_narratives.video_narrative_grounding import vng_expression


class VNGVideo:
  """A video with Video Narrative Grounding annotations."""

  def __init__(
      self,
      video_name: str,
      meta: util.JsonData,
      masks: dict[int, util.JsonData],
      frames_path: Optional[str],
  ):
    self._name = video_name
    self._meta = meta
    self._masks = masks
    self._frames_path = frames_path

    self._expressions = meta['expressions']

  def get_name(self) -> str:
    return self._name

  def __len__(self) -> int:
    return len(self._expressions)

  def __getitem__(self, idx: int) -> vng_expression.VNGExpression:
    if idx >= len(self):
      raise IndexError
    expression = self._expressions[str(idx)]
    return vng_expression.VNGExpression(
        expression, self._meta, self._masks, self._frames_path
    )
