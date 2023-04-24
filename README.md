# MOVi-PyTorch

Conversion of MOVi datasets to PyTorch compatible format, and evaluation code.

## Downloading

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