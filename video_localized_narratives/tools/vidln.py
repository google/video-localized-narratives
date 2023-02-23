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

"""Provides the VideoLocalizedNarrative class for easy use of VidLN data."""

import os
from typing import Optional

from video_localized_narratives.tools import actor_narrative
from video_localized_narratives.tools import frame
from video_localized_narratives.tools import util


class VideoLocalizedNarrative:
  """A Video Localized Narrative annotation for one video with mouse traces."""

  def __init__(self, raw_data: util.JsonData, frames_path: Optional[str]):
    self._raw_data = raw_data
    self._frames_path = frames_path

    self._dataset_id: str = raw_data['dataset_id']
    self._video_id: str = raw_data['video_id']
    self._annotator_id: str = raw_data['annotator_id']
    self._all_keyframe_names: list[str] = raw_data['keyframe_names']
    self._actor_narratives: list[util.JsonData] = raw_data['actor_narratives']
    self._vidln_id: int = raw_data['vidln_id']

  def get_raw_data(self) -> util.JsonData:
    return self._raw_data

  def get_dataset_name(self) -> str:
    return self._dataset_id

  def get_video_name(self) -> str:
    return self._video_id

  def get_annotator_id(self) -> str:
    return self._annotator_id

  def get_all_keyframe_names(self) -> list[str]:
    return self._all_keyframe_names

  def get_actor_names(self) -> list[str]:
    return [n['actor_name'] for n in self._actor_narratives]

  def get_vidln_id(self) -> int:
    return self._vidln_id

  def get_video_frames_root(self) -> Optional[str]:
    return os.path.join(self._frames_path, self.get_video_name())

  def get_all_frames(self) -> list[frame.Frame]:
    return util.get_all_frames(self.get_video_frames_root())

  def get_actor_narratives(self) -> list[actor_narrative.ActorNarrative]:
    return [
        actor_narrative.ActorNarrative(self, actor_data)
        for actor_data in self._actor_narratives
    ]
