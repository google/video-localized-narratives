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

"""Provides the ActorNarrative class for a VidLN narrative of one actor."""

from typing import Any

from video_localized_narratives.tools import frame
from video_localized_narratives.tools import mouse_trace
from video_localized_narratives.tools import util

vidln = Any  # vidln.py imports this module.


class ActorNarrative:
  """A VidLN narrative of one actor."""

  def __init__(
      self, vln: 'vidln.VideoLocalizedNarrative', actor_data: util.JsonData
  ):
    self._vln = vln
    self._actor_data = actor_data

  def get_raw_data(self) -> util.JsonData:
    return self._actor_data

  def get_actor_name(self) -> str:
    return self._actor_data['actor_name'].strip()

  def get_caption(self) -> str:
    return self._actor_data['caption'].strip()

  def get_selected_keyframes(self) -> list[frame.KeyFrame]:
    kf_indices = self._get_selected_keyframe_indices()
    kf_names = self._get_selected_keyframe_names()
    root = self._vln.get_video_frames_root()
    assert len(kf_indices) == len(kf_names)
    return [
        frame.KeyFrame(name, root, None, idx)
        for idx, name in zip(kf_indices, kf_names)
    ]

  def _get_selected_keyframe_indices(self) -> list[int]:
    return self._actor_data['keyframe_selection_indices']

  def _get_selected_keyframe_names(self) -> list[str]:
    all_keyframes = self._vln.get_all_keyframe_names()
    kf_indices = self._get_selected_keyframe_indices()
    return [all_keyframes[kf_index] for kf_index in kf_indices]

  def get_mouse_trace(self) -> mouse_trace.MouseTrace:
    return mouse_trace.MouseTrace(self.get_raw_data())

  def __str__(self) -> str:
    return '<' + self.get_actor_name() + '> ' + self.get_caption()
