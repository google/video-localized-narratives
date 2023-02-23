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

"""Functions to predict an approximate square bounding box from a trace mask."""

import numpy as np

_LEARNED_POLYNOM_COEFS = (0.239715, 1.6737112, -0.42075962)


def estimate_approximate_square_gt_mask(trace_mask: np.ndarray) -> np.ndarray:
  height, width, *_ = trace_mask.shape
  trace_center_yx = _center_of_mass_yx(trace_mask)
  estimated_square_side_length = _predict_side_length(
      trace_mask, trace_center_yx
  )
  return _square_to_mask(
      trace_center_yx, estimated_square_side_length, height, width
  )


def _predict_side_length(
    trace_mask: np.ndarray, center_yx: tuple[int, int]
) -> float:
  """Predict the side length of the approximate square bounding box."""
  trace_square_side_length = _square_side_length(trace_mask, center_yx)

  height, width, *_ = trace_mask.shape
  norm = min(height, width)
  norm_side_length = trace_square_side_length / norm

  # Here, we use a learned quadratic function. For details, see the Video
  # Localized Narratives paper.
  coefs = _LEARNED_POLYNOM_COEFS
  norm_pred = (
      coefs[0]
      + coefs[1] * norm_side_length
      + coefs[2] * norm_side_length * norm_side_length
  )

  pred = norm_pred * norm
  return pred


def _square_side_length(
    trace_mask: np.ndarray, center_yx: tuple[int, int]
) -> float:
  ys, xs = trace_mask.nonzero()
  cy, cx = center_yx
  yl = np.abs(ys - cy).max()
  xl = np.abs(xs - cx).max()
  return 2 * max(yl, xl)


def _center_of_mass_yx(mask: np.ndarray) -> tuple[int, int]:
  ys, xs = mask.nonzero()
  cy = int(round(float(ys.mean())))
  cx = int(round(float(xs.mean())))
  return cy, cx


def _square_to_mask(
    center_yx: tuple[int, int], side_length: float, height: int, width: int
) -> np.ndarray:
  """Represent the square as a segmentation mask."""
  m = np.zeros((height, width))
  yc, xc = center_yx
  y0 = int(round(yc - side_length / 2))
  y1 = int(round(yc + side_length / 2))
  x0 = int(round(xc - side_length / 2))
  x1 = int(round(xc + side_length / 2))
  y0 = max(y0, 0)
  x0 = max(x0, 0)
  y1 = min(y1, height)
  x1 = min(x1, width)
  m[y0:y1, x0:x1] = 1
  return m
