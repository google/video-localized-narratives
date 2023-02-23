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

"""Provides the VNGExpression class."""

from typing import Optional

from video_localized_narratives.tools import frame
from video_localized_narratives.tools import util
from video_localized_narratives.video_narrative_grounding import mask


RED_COLOR = '\x1b[31m'
NORMAL_COLOR = '\x1b[0m'


class VNGExpression:
  """A VNG expression for one object which allows iteration over frames."""

  def __init__(
      self,
      expression: util.JsonData,
      meta: util.JsonData,
      masks: dict[int, util.JsonData],
      frames_path: Optional[str],
  ):
    self._expression = expression
    self._meta = meta
    self._frames_path = frames_path

    ann_id = expression['obj_id']
    self._rles = masks[ann_id]['segmentations']

  def get_description(self) -> str:
    narrative = self.get_narrative()
    return narrative['description']

  def get_description_with_highlighted_noun(self) -> str:
    description = self.get_description()
    narrative = self.get_narrative()
    actor_name = narrative['actor_name']
    start_idx = self._expression['noun_phrase_start_idx']
    end_idx = self._expression['noun_phrase_end_idx']
    return _highlighted_description(description, actor_name, start_idx, end_idx)

  def get_narrative(self) -> util.JsonData:
    actor_idx = self._expression['narrative_actor_idx']
    narrative = self._meta['actor_narratives'][actor_idx]
    return narrative

  def get_all_frames(self) -> list[frame.Frame]:
    assert self._frames_path is not None, 'frames path is not set.'
    return util.get_all_frames(self._frames_path)

  def get_all_frames_and_masks(
      self,
  ) -> list[tuple[frame.Frame, Optional[mask.Mask]]]:
    frames = self.get_all_frames()
    if not frames:
      raise FileNotFoundError(
          f'Did not find frames in {self._frames_path}')
    masks = [mask.Mask(s) if s is not None else s for s in self._rles]
    assert len(frames) == len(masks), (len(frames), len(masks))
    return list(zip(frames, masks))

  def get_annotated_frames_and_masks(
      self,
  ) -> list[tuple[frame.Frame, mask.Mask]]:
    return [(f, m) for f, m in self.get_all_frames_and_masks() if m is not None]

  def get_all_masks(self) -> list[Optional[mask.Mask]]:
    return [mask.Mask(s) if s is not None else None for s in self._rles]


def _highlighted_description(
    description: str, actor_name: str, start_idx: int, end_idx: int
) -> str:
  highlighted = '<' + actor_name + '> ' + description[:start_idx]
  highlighted += RED_COLOR + description[start_idx:end_idx]
  highlighted += NORMAL_COLOR + description[end_idx:]
  return highlighted
