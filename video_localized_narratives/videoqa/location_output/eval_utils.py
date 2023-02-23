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

"""Utilities for evaluation location-output questions."""

import dataclasses
import numpy as np
from pycocotools import mask as cocomask


RECALL_THRESHOLD = 0.5
PRECISION_THRESHOLD = 0.5


@dataclasses.dataclass(frozen=True)
class FrameEvaluationResult:
  recall: float
  precision: float
  recall_criterion: float
  precision_criterion: float
  combined_score: float


def evaluate_result(
    result_mask: np.ndarray,
    approx_gt_square_mask: np.ndarray,
    trace_mask: np.ndarray,
) -> FrameEvaluationResult:
  recall = _eval_recall(result_mask, trace_mask)
  precision = _eval_precision(result_mask, approx_gt_square_mask)
  thresholded_recall = 1.0 if recall >= RECALL_THRESHOLD else 0.0
  thresholded_precision = 1.0 if precision >= PRECISION_THRESHOLD else 0.0
  combined_score = thresholded_recall * thresholded_precision
  return FrameEvaluationResult(
      recall=recall,
      precision=precision,
      recall_criterion=thresholded_recall,
      precision_criterion=thresholded_precision,
      combined_score=combined_score,
  )


def _eval_recall(pred_mask: np.ndarray, trace_mask: np.ndarray) -> float:
  i = np.logical_and(pred_mask, trace_mask).sum()
  a = trace_mask.sum()
  assert a > 0
  ioa = i / a
  return ioa


def _eval_precision(
    pred_mask: np.ndarray, approx_gt_square_mask: np.ndarray
) -> float:
  i = np.logical_and(pred_mask, approx_gt_square_mask).sum()
  a = pred_mask.sum()
  assert approx_gt_square_mask.any()
  if a == 0:
    return 0.0
  ioa = i / a
  return ioa


def box_mask_from_mask(mask: np.ndarray) -> np.ndarray:
  rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
  bbox = cocomask.toBbox(rle)
  x, y, w, h = bbox.round().astype(np.int32)
  image_h, image_w, *_ = mask.shape
  return box_mask_from_bounding_box(image_w, image_h, x, y, w, h)


def box_mask_from_bounding_box(
    image_w: int, image_h: int, x: int, y: int, w: int, h: int) -> np.ndarray:
  box_mask = np.zeros((image_h, image_w), dtype=np.uint8)
  box_mask[y:y+h, x:x+w] = 1
  return box_mask
