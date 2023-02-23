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

"""Utilities to convert a mouse trace to a mask."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from video_localized_narratives.tools import mouse_trace_utils


MATPLOTLIB_POINTS_PER_INCH = 72.0
DEFAULT_TRACE_LINE_WIDTH_PIXELS = 3


def raw_trace_to_mask(
    trace: mouse_trace_utils.RawMouseTrace,
    height: int,
    width: int,
    trace_line_width_pixels: int = DEFAULT_TRACE_LINE_WIDTH_PIXELS,
) -> np.ndarray:
  """Render mouse traces as a np.ndarray mask."""
  # Attention: for rendering mouse traces as masks to work correctly,
  # the matplotlib backend needs to be agg. Otherwise they might be empty.
  # This could cause trouble if the rest of the application uses matplotlib
  # as well.
  prev_backend = matplotlib.get_backend()
  matplotlib.use('agg')

  fig, ax = _make_figure_and_axis(height, width)
  dpi = fig.get_dpi()
  for t in trace:
    xs: list[float] = [trace_el['x'] for trace_el in t]
    ys: list[float] = [trace_el['y'] for trace_el in t]
    _plot(xs, ys, ax, height, width, trace_line_width_pixels, dpi)

  mask = _mask_from_figure(fig)
  ax.clear()
  fig.clear()
  plt.close(fig)

  matplotlib.use(prev_backend)
  return mask


def _array_from_figure(fig: plt.Figure) -> np.ndarray:
  fig.canvas.draw_idle()
  buffer = fig.canvas.get_renderer().buffer_rgba()
  arr = np.frombuffer(buffer, dtype=np.uint8).reshape(
      (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
  )
  return arr


def _mask_from_figure(fig: plt.Figure) -> np.ndarray:
  arr = _array_from_figure(fig)
  mask = (arr[:, :, :3] != 255).any(axis=-1)
  return mask


def _make_figure_and_axis(
    height: int, width: int
) -> tuple[plt.Figure, plt.Axes]:
  """Make matplotlib figure and axis without margins for the specified size."""
  fig = plt.figure()
  # Need to add a small number to avoid problem with rounding down.
  w_inches = (width + 0.1) / fig.get_dpi()
  h_inches = (height + 0.1) / fig.get_dpi()
  fig.set_size_inches((w_inches, h_inches))
  fig.tight_layout(pad=0)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.set_xlim(0, width)
  ax.set_ylim(0, height)
  return fig, ax


def _plot(
    xs: list[float],
    ys: list[float],
    ax: plt.Axes,
    height: int,
    width: int,
    trace_line_width_pixels: int,
    dpi: float,
) -> None:
  """Render the mouse trace points. Used to later convert to a mask."""

  if not xs:
    return
  np_ys = np.array(ys) * height
  np_xs = np.array(xs) * width
  trace_line_width_pts = (
      trace_line_width_pixels * MATPLOTLIB_POINTS_PER_INCH / dpi
  )
  ax.plot(
      np_xs, height - np_ys, linewidth=trace_line_width_pts, antialiased=False
  )
