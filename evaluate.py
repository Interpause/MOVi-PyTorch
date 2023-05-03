"""Evaluation script."""

import logging
from argparse import ArgumentParser

log = logging.getLogger(__name__)

parser = ArgumentParser(description="evaluate MOVi segmentation predictions")

parser.add_argument("name", help="dataset to download (i.e., movi_a)")
parser.add_argument("split", help="dataset split to evaluate on (i.e., validation)")
parser.add_argument(
    "pred_dir",
    help='folder of predictions with same format as "seg" folder created by convert.py',
)
parser.add_argument(
    "-d",
    "--data-dir",
    help='folder to look for converted datasets, defaults to "./out"',
    # TODO: Use __file__ as reference.
    default="./out",
)


def main():
    """Main."""
    args = parser.parse_args()

    from pathlib import Path

    gt_dir = Path(args.data_dir) / args.name / args.split / "seg"
    pred_dir = Path(args.pred_dir)
    assert gt_dir.exists(), "Use convert.py to download dataset first!"

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm

    from movi_pytorch import fg_ari, mIoU

    results = dict(id=[], fg_ari=[], miou=[])

    for vid_dir in tqdm(gt_dir.glob("*/")):
        lbls = [Image.open(p) for p in vid_dir.glob("*.png")]
        preds = [
            Image.open(pred_dir / vid_dir.name / p.name) for p in vid_dir.glob("*.png")
        ]

        if len(lbls) < 1:
            log.warning(f"{vid_dir} is empty!")
            continue

        lbls = torch.stack([torch.from_numpy(np.array(im)) for im in lbls])[None]
        preds = torch.stack([torch.from_numpy(np.array(im)) for im in preds])[None]

        results["id"].append(vid_dir.name)
        results["fg_ari"].append(fg_ari(lbls, preds, max(lbls.max(), preds.max()) + 1))
        results["miou"].append(mIoU(lbls, preds))

    df1 = pd.DataFrame(results)
    df1 = df1.sort_values(by=["id"], ascending=True)
    df1.to_csv(pred_dir / "per-video.csv", sep="\t", index=False)

    df2 = pd.DataFrame([dict(fg_ari=df1.fg_ari.mean(), miou=df1.miou.mean())])
    df2.to_csv(pred_dir / "results.csv", sep="\t", index=False)
    log.info(df2)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    main()
