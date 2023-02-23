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

"""Evaluate a VNG result against the ground truth to get the J&F score."""

from collections.abc import Sequence

from absl import app
from absl import flags
from davis2017 import metrics
import numpy as np
from multiprocessing import Pool

from pathlib import Path
from video_localized_narratives.tools import frame
from video_localized_narratives.tools import util
from video_localized_narratives.video_narrative_grounding import vng_dataset
from video_localized_narratives.video_narrative_grounding import vng_expression
from video_localized_narratives.video_narrative_grounding import vng_video


_META_FILENAME_FLAG = flags.DEFINE_string(
    'meta_filename',
    default=None,
    required=True,
    help='The path the the meta_expressions.json, e.g. "vng/OVIS_VNG/meta_expressions/test/meta_expressions.json"'
)
_EXTRA_MASKS_FILENAME_FLAG = flags.DEFINE_string(
    'extra_masks_filename',
    default=None,
    required=True,
    help='The path to the extra_masks.json file, e.g. "data/vng/OVIS_VNG/extra_masks/test/extra_masks.json"'
)
_ORIG_MASKS_FILENAME_FLAG = flags.DEFINE_string(
    'orig_masks_filename',
    default=None,
    required=True,
    help='The path to the original mask annotations of the dataset, e.g. "data/vng/OVIS_VNG/orig_masks/annotations_train.json"'
)
_RESULT_FOLDER_FLAG = flags.DEFINE_string(
    'result_folder',
    default=None,
    required=True,
    help='The path to the folder with VNG results.'
)
_PARALLEL_FLAG = flags.DEFINE_boolean(
    'parallel',
    default=True,
    help='Whether to run in parallel. Disable for better debuggability.')

_WORKER_COUNT = 12


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  result_folder = _RESULT_FOLDER_FLAG.value
  meta_filename = _META_FILENAME_FLAG.value
  orig_masks_filename = _ORIG_MASKS_FILENAME_FLAG.value
  extra_masks_filename = _EXTRA_MASKS_FILENAME_FLAG.value
  run_parallel = _PARALLEL_FLAG.value

  dataset = vng_dataset.VNGDataset(
      meta_filename=meta_filename, orig_masks_filename=orig_masks_filename,
      extra_masks_filename=extra_masks_filename, frames_path=None)

  jf, j, f, js_by_video_by_exp, fs_by_video_by_exp = evaluate(
      dataset, result_folder, run_parallel
  )
  print('=======')
  print(js_by_video_by_exp)
  print('=======')
  print(fs_by_video_by_exp)
  print('=======')
  print(result_folder)
  print(f'J&F: {jf}')
  print(f'J: {j}')
  print(f'F: {f}')


def evaluate(
    dataset: vng_dataset.VNGDataset, result_folder: str, run_parallel: bool
) -> tuple[float, float, float, util.JsonData, util.JsonData]:
  """Evaluate the VNG result against the VNG ground truth."""
  if run_parallel:
    args = ((v, result_folder) for v in dataset)
    with Pool(processes=_WORKER_COUNT) as pool:
      video_results = pool.starmap(
          evaluate_video, args)
  else:
    video_results = []
    for vid_idx, vng_vid in enumerate(dataset):
      print(vid_idx, '/', len(dataset))
      video_result = evaluate_video(vng_vid, result_folder)
      video_results.append(video_result)
  assert len(dataset) == len(video_results)

  js_by_video_by_exp = {}
  fs_by_video_by_exp = {}
  all_js = []
  all_fs = []
  for video_name, video_result in zip(dataset.get_video_names(), video_results):
    j_by_exp, f_by_exp = video_result
    js_by_video_by_exp[video_name] = j_by_exp
    fs_by_video_by_exp[video_name] = f_by_exp
    all_js.extend(j_by_exp.values())
    all_fs.extend(f_by_exp.values())
  j = float(np.mean(all_js))
  f = float(np.mean(all_fs))
  jf = 0.5 * (j + f)
  return jf, j, f, js_by_video_by_exp, fs_by_video_by_exp


def evaluate_video(
    vng_vid: vng_video.VNGVideo, result_folder: str
) -> tuple[dict[int, float], dict[int, float]]:
  vid_name = vng_vid.get_name()
  j_by_exp_id = {}
  f_by_exp_id = {}
  for exp_id, vng_exp in enumerate(vng_vid):
    j_exp, f_exp = evaluate_expression(vng_exp, result_folder, vid_name, exp_id)
    j_by_exp_id[exp_id] = j_exp
    f_by_exp_id[exp_id] = f_exp
  return j_by_exp_id, f_by_exp_id


def evaluate_expression(
    vng_exp: vng_expression.VNGExpression,
    result_folder: str,
    vid_name: str,
    exp_id: int,
) -> tuple[float, float]:
  """Evaluate the result for one VNG expression. Return J and F scores."""
  pred_mask_filename_by_frame_number = _load_pred_mask_filename_by_frame_number(
      result_folder, vid_name, exp_id
  )
  gt_masks = []
  pred_masks = []

  for frame_number, mask in enumerate(vng_exp.get_all_masks()):
    if mask is None:
      continue
    pred_mask_filename = pred_mask_filename_by_frame_number[frame_number]
    pred_mask = frame.load_img(pred_mask_filename)
    gt_mask = mask.load()
    gt_masks.append(gt_mask)
    pred_masks.append(pred_mask)

  stacked_pred_masks = np.stack(pred_masks)
  stacked_gt_masks = np.stack(gt_masks)
  j_per_frame = metrics.db_eval_iou(stacked_gt_masks, stacked_pred_masks)
  f_per_frame = metrics.db_eval_boundary(stacked_gt_masks, stacked_pred_masks)
  j = j_per_frame.mean()
  f = f_per_frame.mean()
  return j, f


def _load_pred_mask_filename_by_frame_number(
    result_folder: str, vid_name: str, exp_id: int
) -> dict[int, str]:
  masks_dir = Path(result_folder) / vid_name / str(exp_id)
  mask_files = sorted(masks_dir.glob('*.png'))
  if not mask_files:
    raise FileNotFoundError(f'Did not find result files: {masks_dir}/*.png')
  result = {util.frame_number_from_filename(str(f)): str(f) for f in mask_files}

  # Assume contiguous frames.
  for x in range(1, max(result)):
    if x not in result:
      raise ValueError(f'Result pngs are not contiguous: {masks_dir}, {x}')

  # Normalize to 0-indexing.
  if 0 not in result:
    assert 1 in result
    result = {k - 1: v for k, v in result.items()}
  return result


if __name__ == '__main__':
  app.run(main)
