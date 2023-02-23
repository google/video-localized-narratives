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

"""Evaluate a text-output VideoQA result against the ground truth."""

from collections.abc import Sequence

from absl import app
from absl import flags
import sklearn.metrics

from video_localized_narratives.tools import util


_GROUND_TRUTH_JSON_PATH_FLAG = flags.DEFINE_string(
    'gt_json_path',
    default='data/videoqa/text_output/oops_val/qa_text_output.json',
    help='Path to the text-output ground truth data in json format.',
)
_RESULTS_JSON_PATH_FLAG = flags.DEFINE_string(
    'results_path',
    default=None,
    required=True,
    help='Path to the json file with text-output VideoQA results.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gt_json_path = _GROUND_TRUTH_JSON_PATH_FLAG.value
  results_file = _RESULTS_JSON_PATH_FLAG.value
  gt = util.load_json_data(gt_json_path)
  results = util.load_json_data(results_file)
  evaluate(gt, results)


def evaluate(gt: util.JsonData, results: util.JsonData) -> None:
  """Calculate exact match accuracy of result compared against ground truth."""
  gt_answers = []
  predicted_answers = []

  for ann in gt['annotations']:
    for question_answer_pair in ann['qa_pairs']:
      question_id = question_answer_pair['question_id']
      gt_answer = question_answer_pair['answer']
      question_id_short = '_'.join(question_id.split('_')[-3:])
      print(question_id_short, gt_answer)

      if question_id in results:
        prediction = results[question_id]
      elif question_id_short in results:
        prediction = results[question_id_short]
      else:
        raise ValueError(f'Missing result for question: {question_id}')

      gt_answers.append(gt_answer)
      predicted_answers.append(prediction)
  acc = accuracy(gt_answers, predicted_answers)
  print(acc)


def accuracy(targets, predictions):
  return {
      'accuracy (%)': 100 * sklearn.metrics.accuracy_score(targets, predictions)
  }


if __name__ == '__main__':
  app.run(main)
