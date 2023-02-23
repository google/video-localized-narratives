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

"""A mouse trace of a Video Localized Narrative."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from video_localized_narratives.tools import frame
from video_localized_narratives.tools import mouse_trace_to_mask
from video_localized_narratives.tools import mouse_trace_utils
from video_localized_narratives.tools import util


# Use small trace width for generating a precise mask, but use higher value for
# visualization purposes.
DEFAULT_TRACE_WIDTH = mouse_trace_to_mask.DEFAULT_TRACE_LINE_WIDTH_PIXELS
VISUALIZATION_TRACE_WIDTH = 6

DEFAULT_FIGSIZE = (20, 5)


class MouseTrace:
  """A mouse trace of a VidLN, potentially spanning multiple keyframes."""

  def __init__(
      self,
      raw_data: util.JsonData,
      trace: Optional[mouse_trace_utils.RawMouseTrace] = None,
  ):
    self._raw_data = dict(raw_data)
    if trace is not None:
      self._raw_data['traces'] = trace
    self._trace = raw_data['traces']
    self._recording_start_time = raw_data['recording_start_time_ms_since_epoch']

  def is_empty(self) -> bool:
    for t in self._trace:
      for _ in t:
        return False
    return True

  def filter_to_caption_segment(self, start: int, end: int) -> 'MouseTrace':
    filtered_trace = mouse_trace_utils.filter_to_caption_segment(
        self._trace,
        self._raw_data['time_alignment'],
        self._recording_start_time,
        start,
        end,
    )
    return MouseTrace(self._raw_data, filtered_trace)

  def filter_to_keyframe(
      self, keyframe: frame.KeyFrame
  ) -> 'SingleFrameMouseTrace':
    return SingleFrameMouseTrace(self._raw_data, keyframe)

  def visualize(self, keyframes: list[frame.KeyFrame], title: str = '') -> None:
    """Show the mouse trace on the specified keyframes, using matplotlib."""

    imgs = [
        self.filter_to_keyframe(kf).as_overlaid_image(VISUALIZATION_TRACE_WIDTH)
        for kf in keyframes
    ]
    fig, axes = plt.subplots(1, len(imgs), figsize=DEFAULT_FIGSIZE)
    # when we only have a single image, subplots returns a single object
    # instead of an Iterable.
    if len(imgs) < 2:
      axes = [axes]
    for img, axis in zip(imgs, axes):
      axis.imshow(img)
      axis.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


class SingleFrameMouseTrace(MouseTrace):
  """A mouse trace of a VidLN on a single keyframe."""

  def __init__(self, raw_data: util.JsonData, keyframe: frame.KeyFrame):
    filtered_trace = mouse_trace_utils.filter_to_keyframe(
        raw_data['traces'], keyframe.keyframe_idx
    )
    super().__init__(raw_data, filtered_trace)
    self._keyframe = keyframe

  def as_mask(
      self,
      trace_line_width_pixels: int = DEFAULT_TRACE_WIDTH,
      height: Optional[int] = None,
      width: Optional[int] = None,
  ) -> np.ndarray:
    if height is None or width is None:
      img = self._keyframe.load()
      height, width, _ = img.shape
    return mouse_trace_to_mask.raw_trace_to_mask(
        self._raw_data['traces'], height, width, trace_line_width_pixels
    )

  def as_overlaid_image(
      self, trace_line_width_pixels: int = DEFAULT_TRACE_WIDTH
  ) -> np.ndarray:
    img = self._keyframe.load()
    height, width, _ = img.shape
    mask = self.as_mask(trace_line_width_pixels, height, width)
    overlay_color = (0, 255, 0)
    return util.overlay_mask(img, mask, alpha=0.7, overlay_color=overlay_color)
