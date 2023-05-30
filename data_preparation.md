Visit the [project page](https://google.github.io/video-localized-narratives)
to find the download links for our annotations.

Note that we do not provide the raw videos and frames, and you have to download
them according to the instructions of the original datasets:
* [OVIS](https://songbai.site/ovis/)
* [UVO (v1.0)](https://sites.google.com/corp/view/unidentified-video-object/dataset)
* [Oops](https://oops.cs.columbia.edu/data/)
* [Kinetics](https://www.deepmind.com/open-source/kinetics)

You can use this [script](https://drive.google.com/file/d/1C0TKKVIOfaQ_qiVxGuQbMAlXUcTd3B4h/view?usp=sharing)
provided with the UVO dataset, to extract frames from the video files.

We recommend the following structure for the data (not complete):
```
video-localized-narratives/data/
├── vidlns
│   ├── OVIS_train.jsonl
│   ├── UVO_sparse_train.jsonl
│   ├── UVO_sparse_val.jsonl
│   ├── UVO_dense_train.jsonl
│   ├── UVO_dense_val.jsonl
│   ├── oops_train.jsonl
│   ├── oops_val.jsonl
├── frames
│   ├── OVIS_train
│   │   │── 5d24e4ea
│   │   │   │── img_0000001.jpg
│   │   │   │── ...
│   │   │── ...
│   ├── UVO_sparse_train
│   │   │── zxE180Fndow
│   │   │   │── 0.png
│   │   │   │── ...
│   │   │   │── 299.png
│   ├── UVO_sparse_val
│   │   │── ...
│   ├── oops_train
│   │   │── Your Tooth Is Missing - Best Fails of the Week (November 2017) _ FailArmy9
│   │   │   │── 000000.png
│   │   │   │── ...
│   │   │   │── 000249.png
│   │   │── ...
│   ├── oops_val
│   │   │── ...
├── recordings (optional)
│   ├── OVIS_train
│   │   │── 0_0.webm
│   │   │── 0_1.webm
│   │   │── ...
│   ├── UVO_sparse_train
│   │   │── 0_0.webm
│   │   │── ...
├── videos (optional)
│   ├── OVIS_train
│   │   │── 5d24e4ea.mp4
│   │   │── ...
│   ├── ...
├── vng (optional)
│   ├── OVIS_VNG
│   │   │── extra_masks
│   │   │   │── train
│   │   │   │   │── extra_masks.json
│   │   │   │── test
│   │   │   │   │── extra_masks.json
│   │   │── meta_expressions
│   │   │   │── train
│   │   │   │   │── meta_expressions.json
│   │   │   │── test
│   │   │   │   │── meta_expressions.json
│   │   │── orig_masks
│   │   │   │── train
│   │   │   │   │── annotations_train.json (download this from OVIS)
│   │   │   │── test
│   │   │   │   │── annotations_train.json (download this from OVIS, here train is also used for the test set)
│   ├── UVO_VNG
│   │   │── extra_masks
│   │   │   │── train
│   │   │   │   │── extra_masks.json
│   │   │   │── test
│   │   │   │   │── extra_masks.json
│   │   │── meta_expressions
│   │   │   │── train
│   │   │   │   │── meta_expressions.json
│   │   │   │── test
│   │   │   │   │── meta_expressions.json
│   │   │── orig_masks
│   │   │   │── train
│   │   │   │   │── UVO_sparse_train_video.json (download this from UVO)
│   │   │   │── val
│   │   │   │   │── UVO_sparse_val_video.json (download this from UVO)
├── videoqa (optional)
│   ├── text_output
│   ├── location_output
```
Note that `recordings` (the audio files) and `videos` (.mp4 video files) are
optional and mainly used for visualization. The `vng` folder is only necessary,
if you are interested in the Video Narrative Grounding task, and the `videoqa`
folder is only necessary if you are interested in the Video Question-Answering
task.
