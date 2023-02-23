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

"""Provides the LocationOutputQuestion class and tools to load the questions."""

import dataclasses
from typing import Any, Iterator

import numpy as np
from pycocotools import mask as cocomask

from video_localized_narratives.tools import util


@dataclasses.dataclass(frozen=True)
class LocationOutputQuestion:
  video_name: str
  question_hash: str
  question: str
  trace_frame: str
  trace: dict[str, Any]

  def get_trace_mask(self) -> np.ndarray:
    return cocomask.decode(self.trace)


def iterate_location_output_questions(
    gt_json_path: str,
) -> Iterator[LocationOutputQuestion]:
  questions_data_by_video_name = util.load_json_data(gt_json_path)
  for video_name, video_questions_data in questions_data_by_video_name.items():
    yield from iterate_where_questions_for_video(
        video_name, video_questions_data
    )


def iterate_where_questions_for_video(
    video_name: str, video_questions_data: list[dict[str, Any]]
) -> Iterator[LocationOutputQuestion]:
  for d in video_questions_data:
    question_hash = d['question_hash']
    question = d['question']
    trace_frame = d['trace_frame']
    trace = d['trace']
    yield LocationOutputQuestion(
        video_name=video_name,
        question_hash=question_hash,
        question=question,
        trace_frame=trace_frame,
        trace=trace,
    )
