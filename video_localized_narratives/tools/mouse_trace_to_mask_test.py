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

import numpy as np

from video_localized_narratives.tools import mouse_trace_to_mask
from video_localized_narratives.tools import mouse_trace_utils

from absl.testing import absltest
from absl.testing import parameterized


Trace = mouse_trace_utils.RawMouseTrace
TraceElement = mouse_trace_utils.RawMouseTraceElement


class MouseTraceToMaskTest(parameterized.TestCase):

  def test_trace_to_mask_for_empty_trace(self):
    trace: Trace = []
    mask = mouse_trace_to_mask.raw_trace_to_mask(
        trace, height=480, width=640, trace_line_width_pixels=1
    )

    self.assertEqual(mask.shape, (480, 640))
    self.assertFalse(mask.any())

  def test_traces_as_masks_for_line_width1(self):
    height = 480
    width = 640
    py = 42
    px0 = 31
    px1 = px0 + 5
    trace_el0 = _make_trace_element(
        x_absolute=px0, y_absolute=py, width=width, height=height, time=42
    )
    trace_el1 = _make_trace_element(
        x_absolute=px1, y_absolute=py, width=width, height=height, time=43
    )
    trace: Trace = [[trace_el0, trace_el1]]

    mask = mouse_trace_to_mask.raw_trace_to_mask(
        trace, height=height, width=width, trace_line_width_pixels=1
    )
    self.assertEqual(mask.shape, (height, width))

    expected = np.zeros((height, width), dtype=bool)
    expected[py, px0 : px1 + 1] = 1
    self.assertTrue((mask == expected).all())

  @parameterized.named_parameters(
      ('line_width_1', 1),
      ('line_width_2', 2),
      ('line_width_3', 3),
      ('line_width_4', 4),
      ('line_width_5', 5),
      ('line_width_6', 6),
      ('line_width_7', 7),
  )
  def test_traces_as_masks_for_line(self, line_width_px: int):
    height = 1080
    width = 1920
    py = 834
    px0 = 1337
    px1 = px0 + 20
    trace_el0 = _make_trace_element(
        x_absolute=px0, y_absolute=py, width=width, height=height, time=42
    )
    trace_el1 = _make_trace_element(
        x_absolute=px1, y_absolute=py, width=width, height=height, time=43
    )
    trace: Trace = [[trace_el0, trace_el1]]

    mask = mouse_trace_to_mask.raw_trace_to_mask(
        trace, height=height, width=width, trace_line_width_pixels=line_width_px
    )
    self.assertEqual(mask.shape, (height, width))

    # for even values of line_width_px, the line is extended on the left,
    # but not on the right
    over_border_left = line_width_px // 2
    over_border_right = (line_width_px - 1) // 2
    expected = np.zeros((height, width), dtype=bool)
    expected[
        py - over_border_left : py + over_border_right + 1,
        px0 - over_border_left : px1 + over_border_right + 1,
    ] = 1
    self.assertTrue((mask == expected).all())

  def test_traces_as_masks_for_diagonal_line_width_1(self):
    height = 540
    width = 960
    line_width_px = 1
    py0 = 123
    px0 = 234
    line_length = 11
    py1 = py0 + line_length
    px1 = px0 + line_length
    trace_el0 = _make_trace_element(
        x_absolute=px0, y_absolute=py0, width=width, height=height, time=11
    )
    trace_el1 = _make_trace_element(
        x_absolute=px1, y_absolute=py1, width=width, height=height, time=987
    )
    trace: Trace = [[trace_el0, trace_el1]]

    mask = mouse_trace_to_mask.raw_trace_to_mask(
        trace, height=height, width=width, trace_line_width_pixels=line_width_px
    )
    self.assertEqual(mask.shape, (height, width))

    expected_true = np.zeros((height, width), dtype=bool)
    # The line will not be exactly like this, so we only make a rough test this
    # time.
    for l in range(line_length):
      expected_true[py0 + l, px0 + l] = 1
    self.assertTrue(mask[expected_true].all())

    can_be_true_mask = np.zeros((height, width), dtype=bool)
    tolerance = 2
    for l in range(line_length):
      y = py0 + l
      x = px0 + l
      can_be_true_mask[
          y - tolerance : y + tolerance, x - tolerance : x + tolerance
      ] = 1
    has_to_be_false = np.logical_not(can_be_true_mask)
    self.assertFalse(mask[has_to_be_false].any())


def _make_trace_element(
    *, x_absolute: int, y_absolute: int, width: int, height: int, time: int
) -> TraceElement:
  x_rel = x_absolute / width
  y_rel = y_absolute / height
  return {'x': x_rel, 'y': y_rel, 'time_ms_since_epoch': time, 'kf_idx': 2}


if __name__ == '__main__':
  absltest.main()
