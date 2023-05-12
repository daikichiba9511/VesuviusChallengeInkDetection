from __future__ import annotations

from typing import Container

import numpy as np
import torch


def rand_bbox(size: Container[int], lam: float):
    """CutMixのbboxを作成する

    Args:
        size: (B, C, H, W)
        lam: 乱数

    Returns:
        bbox: (left, top, right, bottom)

    Reference:
    [1]
    https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L279:295
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"Invalid lambda: {lam}")

    if len(size) != 4:
        raise ValueError(f"Invalid size: {size}")

    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# cutmix implements using images, and labels
def cutmix(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """CutMixの実装

    Args:
        images: (N, C, H, W)
        labels: (N, H, W)
        alpha: beta分布のパラメータ

    Returns:
        mixed_image: (N, C, H, W)
        mixed_labels: (N, H, W), mask
        lambd: 重み
        rand_idx: 乱数のインデックス

    Reference:
    [1]
    https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L229:L237
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid alpha: {alpha}")

    # generate mixed sample
    # lam = np.random.beta(args.beta, args.beta)
    # rand_index = torch.randperm(input.size()[0]).cuda()
    # target_a = target
    # target_b = target[rand_index]
    # bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    # input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # # adjust lambda to exactly match pixel ratio
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    lam = np.random.beta(alpha, alpha)
    batch_size = images.shape[0]
    rand_idx = torch.randperm(batch_size)

    # ランダムに矩形領域を選ぶ
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    # mixed images
    # images shape: (N, C, H, W)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_idx, :, bbx1:bbx2, bby1:bby2]

    # mixed masks
    # labels shape: (N, H, W)
    labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_idx, bbx1:bbx2, bby1:bby2]
    return images, labels, lam, rand_idx


def mixup(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Args:
        images: (N, C, H, W)
        labels: (N, H, W)
        alpha: beta分布のパラメータ

    Returns:
        mixed_image: (N, C, H, W)
        mixed_labels: (N, H, W), mask
        lambd: 重み
        rand_idx: 乱数のインデックス
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid alpha: {alpha}")

    lam = np.random.beta(alpha, alpha)
    batch_size = images.shape[0]
    rand_idx = torch.randperm(batch_size)

    # (N, C, H, W)
    mixed_image = lam * images + (1 - lam) * images[rand_idx, ...]
    # (N, 1, H, W)
    mixed_labels = lam * labels + (1 - lam) * labels[rand_idx, ...]
    return mixed_image, mixed_labels, lam, rand_idx
