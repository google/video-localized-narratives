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

"""A demonstration for loading and visualizing Video Narrative Grounding data.

This demo uses matplotlib to display images.
"""

from collections.abc import Sequence
import itertools
import os

from absl import app
import matplotlib.pyplot as plt

from video_localized_narratives.tools import util
from video_localized_narratives.video_narrative_grounding import vng_dataset
from video_localized_narratives.video_narrative_grounding import vng_expression
from video_localized_narratives.video_narrative_grounding import vng_video


# Change these paths to where you put your data.
# Note that for OVIS the original masks file contains both the masks for the
# training and the validation set, so you have to specify
# 'annotations_train.json' also for the validation set.
VNG_DATA_ROOT = 'data/vng/'
FRAMES_PATH = 'data/frames/OVIS_train/'

META_FILENAME = os.path.join(
    VNG_DATA_ROOT, 'OVIS_VNG/meta_expressions/train/meta_expressions.json')
EXTRA_MASKS_FILENAME = os.path.join(
    VNG_DATA_ROOT, 'OVIS_VNG/extra_masks/train/extra_masks.json')
ORIG_MASKS_FILENAME = os.path.join(
    VNG_DATA_ROOT, 'OVIS_VNG/orig_masks/annotations_train.json')


FIRST_VIDEO_IDX = 60
N_VIDEOS = 3
MAX_FRAMES_PER_OBJ_TO_SHOW = 3


def visualize_vng(vng_vid: vng_video.VNGVideo) -> None:
  """Visualize Video Narrative Grounding annotations for one video."""
  print(vng_vid.get_name())
  vng_exp: vng_expression.VNGExpression
  for vng_exp in vng_vid:
    visualize_vng_expression(vng_exp)


def visualize_vng_expression(vng_exp: vng_expression.VNGExpression) -> None:
  """Visualize one Video Narrative Grounding expression for one video."""
  print(vng_exp.get_description_with_highlighted_noun())
  for frame, mask in vng_exp.get_annotated_frames_and_masks()[
      :MAX_FRAMES_PER_OBJ_TO_SHOW
  ]:
    img = frame.load()
    loaded_mask = mask.load()
    overlaid = util.overlay_mask(img, loaded_mask, overlay_color=(0, 255, 0))

    print(frame, img.shape, loaded_mask.shape)
    plt.imshow(overlaid)
    plt.show()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset = vng_dataset.VNGDataset(
      meta_filename=META_FILENAME, orig_masks_filename=ORIG_MASKS_FILENAME,
      extra_masks_filename=EXTRA_MASKS_FILENAME, frames_path=FRAMES_PATH)

  vid: vng_video.VNGVideo
  for vid in itertools.islice(
      dataset, FIRST_VIDEO_IDX, FIRST_VIDEO_IDX + N_VIDEOS
  ):
    visualize_vng(vid)
    print('========')


if __name__ == '__main__':
  app.run(main)
