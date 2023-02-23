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

"""Provides the VideoNarrativeGroundingDataset class for loading of VNG data."""

import os
from typing import Any, Union, Optional

from video_localized_narratives.tools import util
from video_localized_narratives.video_narrative_grounding import vng_video


class VNGDataset:
  """A dataset of videos with Video Narrative Grounding annotations.

  Set masks_need_none_padding=True for UVO and False for OVIS

  Usage example:
    dataset = VNGDataset(...)
    for vid in dataset:
      for vng_expression in vid:
        do_something_with(vng_expression)
  """

  def __init__(
      self,
      meta_filename: str,
      orig_masks_filename: str,
      extra_masks_filename: str,
      frames_path: Optional[str],
  ):
    self._meta_data = util.load_json_data(meta_filename)['videos']
    self._video_names = tuple(sorted(self._meta_data.keys()))
    self._frames_path = frames_path

    orig_masks_data = util.load_json_data(orig_masks_filename)
    extra_masks_data = util.load_json_data(extra_masks_filename)
    self._mask_annotation_by_id = _load_mask_annotation_by_id(
        orig_masks_data, extra_masks_data
    )
    # For UVO, the masks need to be padded with None, because they are
    # temporally sparse. For OVIS, no padding is necessary, because the masks
    # are in a dense format.
    # We try to automatically detect UVO here by checking if the field 'ytid'
    # is present.
    masks_need_none_padding = 'ytid' in extra_masks_data['videos'][0]
    if masks_need_none_padding:
      self._pad_sparse_segmentations_with_nones(
          orig_masks_data, extra_masks_data
      )

  def get_video_names(self) -> tuple[str, ...]:
    return self._video_names

  def __getitem__(
      self, video_name_or_idx: Union[int, str]
  ) -> vng_video.VNGVideo:
    if isinstance(video_name_or_idx, int):
      video_name = self._video_names[video_name_or_idx]
    else:
      video_name = video_name_or_idx
    video_meta = self._meta_data[video_name]
    if self._frames_path is None:
      video_frames_path = None
    else:
      video_frames_path = os.path.join(self._frames_path, video_name)
    return vng_video.VNGVideo(
        video_name, video_meta, self._mask_annotation_by_id, video_frames_path
    )

  def __len__(self) -> int:
    return len(self._video_names)

  def _pad_sparse_segmentations_with_nones(
      self, orig_masks_data: util.JsonData, extra_masks_data: util.JsonData
  ) -> None:
    frame_infos_by_video_id = self._load_frame_infos_by_video_id(
        orig_masks_data, extra_masks_data
    )
    for ann in self._mask_annotation_by_id.values():
      vid_id: int = ann['video_id']
      frame_infos = frame_infos_by_video_id[vid_id]
      self._pad_sparse_segmentation_with_nones(ann, frame_infos)

  def _load_frame_infos_by_video_id(
      self, orig_masks_data: util.JsonData, extra_masks_data: util.JsonData
  ) -> dict[int, util.JsonData]:
    all_video_infos = orig_masks_data['videos'] + extra_masks_data['videos']
    video_infos_by_id = {info['id']: info for info in all_video_infos}
    video_infos_by_name = {info['ytid']: info for info in all_video_infos}

    # Add all_frames field to video infos
    for video_name, full_frame_info in self._meta_data.items():
      all_frames = full_frame_info['frames']
      video_infos_by_name[video_name]['all_frames'] = all_frames
    return video_infos_by_id

  def _pad_sparse_segmentation_with_nones(
      self, ann: util.JsonData, frame_infos: util.JsonData
  ) -> None:
    """Pad segmentations with Nones so that we have an entry for every frame."""
    sparse_file_names = frame_infos['file_names']
    if 'all_frames' not in frame_infos:
      return
    all_frames = frame_infos['all_frames']

    sparse_frame_indices = [
        util.frame_number_from_filename(name) for name in sparse_file_names
    ]
    dense_frame_indices = [int(f) for f in all_frames]

    # Remove bboxes and areas because they are also sparse and thus wrong
    # if they are used. However, I think they are not used anyway.
    if 'bboxes' in ann:
      del ann['bboxes']
    if 'areas' in ann:
      del ann['areas']

    sparse_segs = ann['segmentations']
    dense_segs = [None for _ in dense_frame_indices]

    assert len(sparse_frame_indices) == len(sparse_segs), (
        sparse_frame_indices,
        sparse_segs,
    )
    for idx, seg in zip(sparse_frame_indices, sparse_segs):
      assert dense_frame_indices[idx] == idx, (dense_frame_indices, idx)
      dense_segs[idx] = seg

    ann['segmentations'] = dense_segs


def _load_mask_annotation_by_id(
    orig_masks_data: util.JsonData,
    extra_masks_data: util.JsonData,
) -> dict[int, util.JsonData]:
  ann_by_id_1 = _load_mask_annotation_by_id_single_file(orig_masks_data)
  ann_by_id_2 = _load_mask_annotation_by_id_single_file(extra_masks_data)
  assert set(ann_by_id_1).isdisjoint(ann_by_id_2)
  ann_by_id = {**ann_by_id_1, **ann_by_id_2}
  return ann_by_id


def _load_mask_annotation_by_id_single_file(
    masks_data: util.JsonData,
) -> dict[int, Any]:
  ann_by_id = {}
  for ann in masks_data['annotations']:
    id_ = ann['id']
    if id_ in ann_by_id:
      print('warning, found duplicate annotation id', id_)
      continue
    ann_by_id[id_] = ann
  return ann_by_id
