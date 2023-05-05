# MOVi-PyTorch

Conversion of MOVi tfrecord datasets to PyTorch-friendly format, and FG-ARI & mIoU evaluation code.

## Installation

```sh
git clone https://github.com/Interpause/MOVi-PyTorch.git
cd MOVi-PyTorch
pip install -r requirements.txt
```

## Format

- `rgb`: Folder containing JPEG-encoded video frames per video.
- `seg`: Folder containing PNG-encoded instance segmentations (using color palette) per video.

## Conversion

```sh
python convert.py movi_a
```

- Use `--help` to see all options.
- `--data-dir` defaults to `gs://kubric-public/tfds`, meaning files will be downloaded at runtime.
- Optionally, they can be downloaded beforehand, see [below](#persistent-download).
  - Afterwards, set `--data-dir` to `./data`.

## Evaluation

```sh
python evaluate.py movi_a validation <pred_dir>
```

- Use `--help` to see all options.
- `pred_dir` should have the same format and structure as `seg` folder.
- `per-video.csv` and `results.csv` containing FG-ARI & mIoU scores will be written to `pred_dir`.

## Persistent Download

See <https://cloud.google.com/storage/docs/gsutil_install> for how to install `gsutil`.

1. Determine dataset and image size wanted:

```sh
DATASET=movi_a
SIZE=256x256
mkdir -p data/${DATASET}/${SIZE}
```

2. Download entire dataset:

```sh
gsutil -m cp -nr gs://kubric-public/tfds/${DATASET}/.config data/${DATASET}/
gsutil -m cp -nr gs://kubric-public/tfds/${DATASET}/${SIZE}/1.0.0 data/${DATASET}/${SIZE}/
```

3. Or copy specific split:

```sh
SPLIT=validation
mkdir -p data/${DATASET}/${SIZE}/1.0.0
gsutil -m cp -nr gs://kubric-public/tfds/${DATASET}/.config data/${DATASET}/
gsutil -m cp -nr \
    gs://kubric-public/tfds/${DATASET}/${SIZE}/1.0.0/dataset_info.json \
    gs://kubric-public/tfds/${DATASET}/${SIZE}/1.0.0/features.json \
    gs://kubric-public/tfds/${DATASET}/${SIZE}/1.0.0/*.labels.txt \
    data/${DATASET}/${SIZE}/1.0.0/
gsutil -m cp -nr gs://kubric-public/tfds/${DATASET}/${SIZE}/1.0.0/${DATASET}-${SPLIT}* data/${DATASET}/${SIZE}/1.0.0/
```
