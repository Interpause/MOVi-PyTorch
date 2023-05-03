"""Dataset conversion script."""

import logging
from argparse import ArgumentParser

log = logging.getLogger(__name__)

parser = ArgumentParser(description="convert MOVi tfrecord files to images")

parser.add_argument("name", help="dataset to download (i.e., movi_a)")
parser.add_argument(
    "-d",
    "--data-dir",
    help='data_dir for tfds.load(), defaults to "gs://kubric-public/tfds"',
    default="gs://kubric-public/tfds",
)
parser.add_argument(
    "-p", "--split", help="split for tfds.load(), defaults to None", default=None
)
parser.add_argument(
    "-o",
    "--out-dir",
    help='directory to save converted images to, defaults to "./out"',
    default="./out",
)


def main():
    """Main."""
    args = parser.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Prevent tensorflow from claiming GPU as it is unneeded.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    log.info(
        f"Loading {args.name}{'' if args.split is None else f'[{args.split}]'} from {args.data_dir}..."
    )

    datasets = tfds.load(name=args.name, split=args.split, data_dir=args.data_dir)
    if not isinstance(datasets, dict):
        datasets = {args.split: datasets}

    from pathlib import Path

    from tqdm.contrib import tenumerate

    from movi_pytorch import save_image

    for split, ds in datasets.items():
        log.info(f"Converting split {split}...")
        out_dir = Path(args.out_dir) / args.name / str(split)

        for i, d in tenumerate(tfds.as_numpy(ds)):
            vid_pth = out_dir / "rgb" / f"{i:05d}"
            lbl_pth = out_dir / "seg" / f"{i:05d}"

            # TODO: Use ThreadExecutorPool to speed up writes.
            for n, (im, lbl) in enumerate(zip(d["video"], d["segmentations"])):
                save_image(im, vid_pth / f"{n:05d}.jpg", palette=None)
                save_image(lbl.squeeze(-1), lbl_pth / f"{n:05d}.png")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    main()
