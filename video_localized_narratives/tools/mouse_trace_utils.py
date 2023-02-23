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

"""Low-level utilities for handling mouse traces."""

import itertools
from typing import Any, Callable, Iterator

from video_localized_narratives.tools import util


# The below data structures directly correspond to the json data for VidLNs,
# that's why they are called 'Raw'. We recommend you to use the wrapper classes
# instead.

# An element of a mouse trace with the following keys:
# x: float between 0 and 1 indicating the x coordinate inside the keyframe.
# y: float between 0 and 1 indicating the y coordinate inside the keyframe.
# time_ms_since_epoch: absolute time stamp in milliseconds, when the trace was
# recorded.
# kf_idx: the index of the keyframe the trace was painted on (int).
# E.g.
# {
#     'x': 0.24,
#     'y': 0.14,
#     'time_ms_since_epoch': 100,
#     'kf_idx': 2
# }
RawMouseTraceElement = dict[str, Any]

# A list of RawMouseTraceElements is a part of a RawMouseTrace.
RawMouseTracePart = list[RawMouseTraceElement]

# A list of RawMouseTracePart is a RawMouseTrace.
RawMouseTrace = list[RawMouseTracePart]

# Provides a time window for one word of a VidLN caption.
# e.g.
# {
#   'end_ms': 34340,
#   'referenced_word': 'grey-red',
#   'referenced_word_end_idx': 104,
#   'referenced_word_start_idx': 96,
#   'start_ms': 33620
# }
# start_ms and end_ms specify the time window in milliseconds.
# referenced_word_start_idx and referenced_word_end_idx are indices into the
# caption such that
# caption[referenced_word_start_idx:referenced_word_end_idx] == referenced_word.
AlignmentElement = dict[str, Any]

# Provides time-stamps for the words of the VidLN caption.
TimeAlignment = list[AlignmentElement]

# A function used to filter a mouse trace.
TracePredicate = Callable[[RawMouseTraceElement], bool]


def filter_to_caption_segment(
    trace: RawMouseTrace,
    alignment: TimeAlignment,
    recording_start_time: int,
    caption_start: int,
    caption_end: int,
) -> RawMouseTrace:
  """Extract the mouse trace segment for caption[caption_start:caption_end]."""

  sorted_alignment: TimeAlignment = sorted(
      alignment, key=lambda el: el['start_ms']
  )
  matching_alignment: TimeAlignment = [
      el
      for el in sorted_alignment
      if _has_overlap_with_alignment_element(el, caption_start, caption_end)
  ]
  if not matching_alignment:
    return []

  start_time = matching_alignment[0]['start_ms']
  last = matching_alignment[-1]
  end_time = _extended_end_time(last, sorted_alignment)

  return _filter_trace_by_relative_time(
      trace, recording_start_time, start_time, end_time
  )


def _has_overlap_with_alignment_element(
    alignment_element: util.JsonData, caption_start: int, caption_end: int
) -> bool:
  alignment_start: int = alignment_element['referenced_word_start_idx']
  alignment_end: int = alignment_element['referenced_word_end_idx']
  intersection_start = max(caption_start, alignment_start)
  intersection_end = min(caption_end, alignment_end)
  return intersection_end > intersection_start


def _extended_end_time(
    alignment_element: util.JsonData, sorted_alignment: TimeAlignment
) -> int:
  """Extend end time until the start of the next word."""
  idx = sorted_alignment.index(alignment_element)
  if idx == len(sorted_alignment) - 1:
    return alignment_element['end_ms'] + 99999999
  next_word = sorted_alignment[idx + 1]
  end = next_word['start_ms']
  return end


def filter_to_keyframe(
    trace: RawMouseTrace, keyframe_idx: int
) -> RawMouseTrace:
  return _filter_trace(
      trace, lambda trace_el: trace_el['kf_idx'] == keyframe_idx
  )


def _filter_trace(
    trace: RawMouseTrace, predicate: TracePredicate
) -> RawMouseTrace:
  return list(_filter_and_split_trace_by_predicate(trace, predicate))


def _filter_trace_by_relative_time(
    trace: RawMouseTrace,
    recording_start_time: int,
    start_time: int,
    end_time: int,
) -> RawMouseTrace:
  """Filter mouse traces to only lie within the specified time span."""
  # start_time and end_time are relative to the beginning of the recording.
  absolute_start_time = start_time + recording_start_time
  absolute_end_time = end_time + recording_start_time

  def _is_in_time_interval(trace_el: RawMouseTraceElement) -> bool:
    return (
        absolute_start_time
        <= trace_el['time_ms_since_epoch']
        <= absolute_end_time
    )

  return list(_filter_and_split_trace_by_predicate(trace, _is_in_time_interval))


def _filter_and_split_trace_by_predicate(
    trace: RawMouseTrace, pred: TracePredicate
) -> Iterator[RawMouseTracePart]:
  """Keep only traces which fulfill the predicate.

  Args:
    trace: the mouse trace to split.
    pred: the predicate used to filter and split.

  Yields:
    The filtered and split mouse trace.

  Splits the trace if pred changes inside it.
  """
  for t in trace:
    yield from _filter_and_split_trace_part_by_predicate(t, pred)


def _filter_and_split_trace_part_by_predicate(
    trace_part: RawMouseTracePart, pred: TracePredicate
) -> Iterator[RawMouseTracePart]:
  """Keep only trace elements which fulfill the predicate.

  Args:
    trace_part: The mouse trace part to filter and split.
    pred: The predicate used to filter and split.

  Yields:
    Mouse trace parts generated by filtering and splitting the input trace part.

  The trace part is split if pred changes, e.g. if pred is first True, then
  False, and then True again.
  """
  for k, group in itertools.groupby(trace_part, key=pred):
    if k:
      yield list(group)
