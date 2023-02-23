# Installation

You need Python 3.9 or higher.

First clone the repository and enter the cloned directory:

```bash
git clone https://github.com/google/video-localized-narratives.git
cd video-localized-narratives
```
You might want to add the `video-localized-narratives` folder to your `PYTHONPATH`.
To do this, while being in the `video-localized-narratives` folder, run
```bash
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

Install the requirements
```bash
pip install -r requirements.txt
```

## Video Narrative Grounding
(Only) for the Video Narrative Grounding Evaluation, you also need the [DAVIS 2017 toolkit](https://github.com/davisvideochallenge/davis2017-evaluation) which you should be able to install like this
```bash
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation
python setup.py install
```
If this does not work, please follow the instructions from the [DAVIS 2017 toolkit](https://github.com/davisvideochallenge/davis2017-evaluation) directly.
