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

""""Evaluate a location-output VideoQA result against the ground truth.

The evaluation is based on bounding boxes, but for convenience we represent the
boxes as masks to calculate the evaluation measures.
"""

from collections.abc import Sequence

import os
from absl import app
from absl import flags
import numpy as np
from multiprocessing import Pool

from video_localized_narratives.tools import frame
from video_localized_narratives.videoqa.location_output import eval_utils
from video_localized_narratives.videoqa.location_output import location_output_question
from video_localized_narratives.videoqa.location_output import square_prediction


GROUND_TRUTH_JSON_PATH_FLAG = flags.DEFINE_string(
    'gt_json_path',
    default='data/videoqa/location_output/oops_val/qa_location_output.json',
    help='Path to the location-output ground truth data in json format',
)
RESULT_FOLDER_FLAG = flags.DEFINE_string(
    'result_folder',
    default=None,
    required=True,
    help='Path to the folder with location-output VideoQA results in the form '
         'of png files for each question_hash.',
)
PARALLEL_FLAG = flags.DEFINE_boolean(
    'parallel',
    default=True,
    help='Whether to run in parallel. Disable for better debuggability.',
)

WORKER_COUNT = 12


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gt_json_path = GROUND_TRUTH_JSON_PATH_FLAG.value
  results_folder = RESULT_FOLDER_FLAG.value
  parallel_flag = PARALLEL_FLAG.value
  evaluate(gt_json_path, results_folder, parallel_flag)


def evaluate(
    gt_json_path: str, results_folder: str, parallel_flag: bool
) -> None:
  """Evaluate a location-output VideoQA result against the ground truth."""
  questions = list(location_output_question.iterate_location_output_questions(
      gt_json_path
  ))
  if parallel_flag:
    with Pool(processes=WORKER_COUNT) as pool:
      args = ((question, results_folder) for question in questions)
      question_results = pool.starmap(
          eval_question, args)
  else:
    question_results = []
    for idx, question in enumerate(questions):
      print(idx, '/', len(questions))
      question_result = eval_question(question, results_folder=results_folder)
      question_results.append(question_result)

  measures = eval_utils.FrameEvaluationResult.__annotations__.keys()
  for m in measures:
    m_scores = [getattr(r, m) for r in question_results]
    m_mean = np.mean(m_scores)
    print(f'{m}: {m_mean:.1%}')


def eval_question(
    question: location_output_question.LocationOutputQuestion,
    results_folder: str,
) -> eval_utils.FrameEvaluationResult:
  result_mask = load_result_box_mask(question, results_folder)

  trace_mask = question.get_trace_mask()
  approx_gt_square_mask = square_prediction.estimate_approximate_square_gt_mask(
      trace_mask
  )
  return eval_utils.evaluate_result(
      result_mask, approx_gt_square_mask, trace_mask
  )


def load_result_box_mask(
    question: location_output_question.LocationOutputQuestion,
    results_folder: str,
) -> np.ndarray:
  """Load the result bounding box represented as a mask.

  Args:
    question: the location-output question for which we load the result mask.
    results_folder: A string pointing to the folder in which the results are
      stored.

  Returns:
    The result bounding box represented as a mask (np.ndarray).

  Here, we load a mask from a .png file and then fill it to it's enclosing
  bounding box, such that we have a bounding box represented as a mask.

  Instead, you can change this function tool load the coordinates of a bounding
  box and then convert it to a bounding box mask using
  eval_utils.box_mask_from_bounding_box.
  """
  res_mask_filename = os.path.join(
      results_folder,
      question.video_name,
      question.question_hash,
      question.trace_frame,
  )
  mask = (frame.load_img(res_mask_filename) > 0).astype(np.uint8)
  # Usually we convert the mask to a box and evaluate the box.
  # Note that the box is still represented as a mask.
  return eval_utils.box_mask_from_mask(mask)


if __name__ == '__main__':
  app.run(main)
