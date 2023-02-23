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

from video_localized_narratives.tools import mouse_trace_utils
from absl.testing import absltest


class MouseTraceUtilsTest(absltest.TestCase):

  def test_noun_phrase_has_overlap_with_alignment_element(self):
    # alignment has "grey-red", while the noun phrase has "gray"
    alignment_el = {
        'end_ms': 34340,
        'referenced_word': 'grey-red',
        'referenced_word_end_idx': 104,
        'referenced_word_start_idx': 96,
        'start_ms': 33620,
    }
    wrong_alignment_el = {
        'end_ms': 3490,
        'referenced_word': 'background',
        'referenced_word_end_idx': 17,
        'referenced_word_start_idx': 7,
        'start_ms': 2710,
    }
    wrong_alignment_el2 = {
        'referenced_word_end_idx': 90,
        'referenced_word_start_idx': 96,
    }
    # The transcription is not actually used, but it is helpful to understand
    # the test.
    # transcription = (
    #    'In the background, there is a grey road with white lines,'
    #    ' a green grassland with brick borders, grey-red houses, '
    #    'green trees with white-painted trunks, green bushes, a '
    #    'black light pole, a brown banner, a grey-yellow dustbin, '
    #    'and a sky.'
    # )
    start = 96
    end = 100
    self.assertTrue(
        mouse_trace_utils._has_overlap_with_alignment_element(
            alignment_el, start, end
        )
    )
    self.assertFalse(
        mouse_trace_utils._has_overlap_with_alignment_element(
            wrong_alignment_el, start, end
        )
    )
    self.assertFalse(
        mouse_trace_utils._has_overlap_with_alignment_element(
            wrong_alignment_el2, start, end
        )
    )

  def test_extended_end_time_from_alignment(self):
    al0 = _make_alignment_element(
        start_ms=100, end_ms=200, start_idx=0, end_idx=5
    )
    al1 = _make_alignment_element(
        start_ms=250, end_ms=300, start_idx=6, end_idx=10
    )
    al2 = _make_alignment_element(
        start_ms=320, end_ms=400, start_idx=12, end_idx=20
    )
    sorted_alignment = [al0, al1, al2]

    self.assertEqual(
        mouse_trace_utils._extended_end_time(al0, sorted_alignment),
        250,
    )
    self.assertEqual(
        mouse_trace_utils._extended_end_time(al1, sorted_alignment),
        320,
    )
    self.assertGreaterEqual(
        mouse_trace_utils._extended_end_time(al2, sorted_alignment),
        9999,
    )


def _make_alignment_element(
    *, start_ms: int, end_ms: int, start_idx: int, end_idx: int
) -> mouse_trace_utils.AlignmentElement:
  return {
      'start_ms': start_ms,
      'end_ms': end_ms,
      'referenced_word_start_idx': start_idx,
      'referenced_word_end_idx': end_idx,
  }


if __name__ == '__main__':
  absltest.main()
