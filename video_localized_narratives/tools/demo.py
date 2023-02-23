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

"""A demonstration for loading and visualizing VidLN data.

This demo uses matplotlib to display images.
"""

from collections.abc import Sequence
import itertools

from absl import app

from video_localized_narratives.tools import actor_narrative
from video_localized_narratives.tools import frame
from video_localized_narratives.tools import vidln_dataset
from video_localized_narratives.tools import mouse_trace
from video_localized_narratives.tools import vidln


# Change these paths to load a different dataset.
JSONL_PATH = 'data/vidlns/OVIS_train_sample.jsonl'
FRAMES_PATH = 'data/frames/OVIS_train/'

N_VIDEOS = 2
# This list is not exhaustive. For these words we usually do not want to process
# or show the mouse trace.
STOP_WORDS = [
    'a',
    'an',
    'of',
    'the',
    'to',
    'and',
    'is',
    'are',
    'on',
    'at',
    'for',
    'with',
    'there',
    'in',
    'then',
    'by',
    'but',
    'suddenly',
]


def visualize_mouse_trace_segments(
    keyframes: list[frame.KeyFrame], caption: str, trace: mouse_trace.MouseTrace
) -> None:
  """Visualize the mouse trace segments for each non-stop word."""
  sp = caption.split()
  postprocessed = [w.replace('.', '').replace(',', '') for w in sp]
  filtered = [w for w in postprocessed if w.lower() not in STOP_WORDS]
  start = 0
  for word in filtered:
    start = caption.find(word, start)
    end = start + len(word)
    trace_seg = trace.filter_to_caption_segment(start, end)
    print(f'Visualizing traces for "{word}"...')
    trace_seg.visualize(keyframes, title=f'{caption}\nTraces for "{word}"')


def visualize_actor_narrative(
    narrative: actor_narrative.ActorNarrative,
) -> None:
  """Visualize an actor narrative together with its mouse traces."""
  print(narrative)

  keyframes = narrative.get_selected_keyframes()
  print(keyframes)

  trace = narrative.get_mouse_trace()
  print('Visualizing all traces...')
  trace.visualize(keyframes, title=f'{narrative.get_caption()}\nAll traces')

  caption = narrative.get_caption()
  visualize_mouse_trace_segments(keyframes, caption, trace)


def visualize_narrative(vln: vidln.VideoLocalizedNarrative) -> None:
  print(vln.get_video_name())

  all_frames = vln.get_all_frames()
  if not all_frames:
    raise FileNotFoundError(
        f'Did not find frames in {vln.get_video_frames_root()}')
  print(len(all_frames), all_frames[0].load().shape)

  for narrative in vln.get_actor_narratives():
    print('--------')
    visualize_actor_narrative(narrative)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset = vidln_dataset.VideoLocalizedNarrativeDataset(
      JSONL_PATH, FRAMES_PATH)
  for vln in itertools.islice(dataset, N_VIDEOS):
    visualize_narrative(vln)
    print('========')


if __name__ == '__main__':
  app.run(main)
