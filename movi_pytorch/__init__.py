"""Misc utils."""

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import jaccard_score
from torch.nn.functional import one_hot

# COCO color palette; There isn't an included color palette.
COLOR_PALETTE = [
    (0, 0, 0),
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
    (255, 77, 255),
    (0, 226, 252),
    (182, 182, 255),
    (0, 82, 0),
    (120, 166, 157),
    (110, 76, 0),
    (174, 57, 255),
    (199, 100, 0),
    (72, 0, 118),
    (255, 179, 240),
    (0, 125, 92),
    (209, 0, 151),
    (188, 208, 182),
    (0, 220, 176),
    (255, 99, 164),
    (92, 0, 73),
    (133, 129, 255),
    (78, 180, 255),
    (0, 228, 0),
    (174, 255, 243),
    (45, 89, 255),
    (134, 134, 103),
    (145, 148, 174),
    (255, 208, 186),
    (197, 226, 255),
    (171, 134, 1),
    (109, 63, 54),
    (207, 138, 255),
    (151, 0, 95),
    (9, 80, 61),
    (84, 105, 51),
    (74, 65, 105),
    (166, 196, 102),
    (208, 195, 210),
    (255, 109, 65),
    (0, 143, 149),
    (179, 0, 194),
    (209, 99, 106),
    (5, 121, 0),
    (227, 255, 205),
    (147, 186, 208),
    (153, 69, 1),
    (3, 95, 161),
    (163, 255, 0),
    (119, 0, 170),
    (0, 182, 199),
    (0, 165, 120),
    (183, 130, 88),
    (95, 32, 0),
    (130, 114, 135),
    (110, 129, 133),
    (166, 74, 118),
    (219, 142, 185),
    (79, 210, 114),
    (178, 90, 62),
    (65, 70, 15),
    (127, 167, 115),
    (59, 105, 106),
    (142, 108, 45),
    (196, 172, 0),
    (95, 54, 80),
    (128, 76, 255),
    (201, 57, 1),
    (246, 0, 122),
    (191, 162, 208),
]


def save_image(
    image: np.ndarray,
    path: Path,
    palette: List[int] = np.array(COLOR_PALETTE).flatten().tolist(),
):
    """Save image with optional palette.

    Ensures parent directory exists before saving image.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    if palette:
        im = Image.fromarray(image.astype(np.uint8), mode="P")
        im.putpalette(palette, "RGB")
    else:
        im = Image.fromarray(image.astype(np.uint8), mode="RGB")
    im.save(path, optimize=True)


def fg_ari(gt: torch.Tensor, pred: torch.Tensor, n_cls: int):
    """Calculate foreground Adjusted Rand Index.

    Adapted from https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py#L111.

    Args:
        gt (torch.Tensor): BTHW ground truth.
        pred (torch.Tensor): BTHW prediction.
        n_cls (int): Number of classes.

    Returns:
        torch.Tensor: FG-ARI scores for each item in batch.
    """
    gt_oh = one_hot(gt.long(), n_cls).float()
    pred_oh = one_hot(pred.long(), n_cls).float()

    # Ignore background
    gt_oh = gt_oh[..., 1:]

    N = torch.einsum("bthwn, bthwm -> bnm", gt_oh, pred_oh)
    A = N.sum(dim=-1)
    B = N.sum(dim=-2)
    n_pts = A.sum(dim=1)

    ridx = torch.sum(N * (N - 1), dim=[1, 2])
    aidx = torch.sum(A * (A - 1), dim=1)
    bidx = torch.sum(B * (B - 1), dim=1)
    e_ridx = aidx * bidx / torch.clamp(n_pts * (n_pts - 1), 1)
    m_ridx = (aidx + bidx) / 2
    denom = m_ridx - e_ridx
    ari = (ridx - e_ridx) / denom

    return ari.where(~denom.isclose(torch.tensor(0.0)), 1.0)


def mIoU(gt, pred, match=False):
    """Calculate mean IoU/Jaccard.

    Args:
        gt (torch.Tensor): BTHW ground truth.
        pred (torch.Tensor): BTHW prediction.
        n_cls (int): Number of classes.

    Returns:
        torch.Tensor: mIoU scores for each item in batch.
    """
    assert match == False, "Matching not supported yet."

    miou = []
    for x, y in zip(pred, gt):
        x = x.flatten()
        y = y.flatten()
        iou = jaccard_score(y, x, average=None)
        miou.append(np.mean(iou))

    return torch.tensor(miou).float()
