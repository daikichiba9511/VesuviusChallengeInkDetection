"""exp057

add cls Head

Ref:
[1] https://www.kaggle.com/code/hengck23/lb0-68-one-fold-stacked-unet
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import ttach as tta
from albumentations.pytorch import ToTensorV2
from einops import rearrange, reduce, repeat
from loguru import logger
from PIL import Image
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock, UnetDecoder
from timm.data.loader import PrefetchLoader
from timm.models.resnet import resnet10t, resnet34d
from torch.utils.data import DataLoader, Dataset

import wandb

IS_TRAIN = not Path("/kaggle/working").exists()
MAKE_SUB: bool = False
SKIP_TRAIN = False

if IS_TRAIN:
    from src.train_utils import (
        EarlyStopping,
        get_alb_transforms,
        get_loss,
        get_optimizer,
        get_scheduler,
        seed_everything,
        train_per_epoch,
        tta_rotate,
        valid_per_epoch,
    )

dbg = logger.debug

# ssl._create_default_https_context = ssl._create_unverified_context
warnings.simplefilter("ignore")

logger.info(f"Meta Config: IS_TRAIN={IS_TRAIN}," + f"MAKE_SUB={MAKE_SUB}," + f"SKIP_TRAIN={SKIP_TRAIN}")


if IS_TRAIN:
    ROOT_DIR = Path(".")
    INPUT_DIR = ROOT_DIR / "input"
    DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"
    OUTPUT_DIR = ROOT_DIR / "output"
    CP_DIR = OUTPUT_DIR
else:
    ROOT_DIR = Path("..")
    INPUT_DIR = ROOT_DIR / "input"
    DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
    OUTPUT_DIR = Path(".")
    CP_DIR = Path("/kaggle/input/ink-model")


@dataclass
class CFG:
    exp_name: str = "exp057_7_stackedUnet"
    mode = ["train", "test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state: int = 42
    num_workers: int = mp.cpu_count()

    # --- Train Config
    n_fold: int = 5
    epoch: int = 15
    batch_size: int = 16
    valid_batch_size: int = 16
    train_fold: list[int] = field(default_factory=lambda: [1])
    # train_fold: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    grad_accum: int = 4
    patience: int = 10
    is_second_stage: bool = False
    use_amp: bool = True
    loss: str = "BCESoftDiceLoss"
    # loss: str = "SoftBCE"
    """BCESoftDiceLoss, DiceLoss, BCEDiceLoss, BCEWithLogitsLoss, TverskyLoss, SoftBCE"""
    bce_weight: float = 0.5  # BCEWithLogitsLossのweight

    weight_cls: float = 0.1
    aux_params: dict[str, Any] = field(default_factory=lambda: {"classes": 1, "dropout": 0.5})

    optimizer: str = "AdamW"
    """RAdam, AdamW"""
    weight_decay: float = 1e-6

    use_diff_lr: bool = False
    lr: float = 3e-5
    # lr: float = 1e-5
    # encoder_lr: float = 1e-4
    # decoder_lr: float = 1e-3

    schedule_per_epoch: bool = True
    scheduler: str = "GradualWarmupScheduler"
    """OneCycleLR, CosineAnnealingLR, GradualWarmupScheduler"""
    T_max: int = 4
    # max_lr: float = encoder_lr

    max_grad_norm: float = 1000.0

    start_awp_epoch: int = 12
    """epoch > start_awpでawpを使う"""
    start_epoch: int = 10
    adv_lr: float = 5e-7
    adv_eps: int = 3
    adv_step: int = 1

    start_freaze_model_epoch: int = epoch
    freeze_keys: list[str] = field(default_factory=lambda: ["encoder1", "encoder2"])

    pretrained: bool = True
    "pretrained modelを使うかどうか"
    resume_train: bool = False
    resume_weights: list[str | Path] = field(
        default_factory=lambda: [
            "output/exp055_stackedUnet/checkpoint_1.pth",
            "output/exp055_stackedUnet/checkpoint_2.pth",
            "output/exp055_stackedUnet/checkpoint_3.pth",
            "output/exp055_stackedUnet/checkpoint_4.pth",
            "output/exp055_stackedUnet/checkpoint_5.pth",
        ]
    )

    # --- Data Config
    fragment_z: tuple[int, int] = (28, 38)
    """should be 0 ~ 65, [z_min, z_max)"""
    fragment_depth: int = fragment_z[1] - fragment_z[0]
    crop_size: int = 384
    """same as tile size."""
    crop_fade: int = 56
    """画像の端の部分を切り取るときに、端の部分をfadeさせることで、画像の端の部分の情報を失わないようにする。
    """
    crop_depth: int = 6
    """どのスライスの高さでcropするか"""
    # -- train
    # stride: int = 224 // 4
    stride: int = crop_size // 4
    """cropするときのstride"""
    # -- test
    # stride: int = crop_size // 2
    # """cropするときのstride"""

    positive_sample_only: bool = False
    """positive sample(inkを含んでるcrop)のみを使うかどうか"""

    mixup: bool = True
    mixup_prob: float = 0.75
    mixup_alpha: float = 0.1

    cutmix: bool = True
    cutmix_prob: float = 0.75
    cutmix_alpha: float = 0.1

    label_noise: bool = True
    label_noise_prob: float = 0.5

    train_compose = [
        A.OneOf(
            [
                A.Resize(crop_size, crop_size),
                A.RandomResizedCrop(crop_size, crop_size, scale=(0.5, 1.8), ratio=(0.5, 1.5)),
            ],
        ),
        A.OneOf(
            [
                A.RandomRotate90(always_apply=True),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=(0.5, 1.8),
                    rotate_limit=360,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        ),
        # A.OneOf([
        # A.RandomBrightnessContrast(p=0.5),
        # A.RandomContrast(limit=0.5, p=0.5),
        # ])
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.RandomContrast(limit=0.5, p=0.5),
            ]
        ),
        # A.Downscale(scale_min=0.5, scale_max=0.95, p=0.5),
        # A.CLAHE(p=0.75),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.2,
        ),
        # 歪み系
        # A.OpticalDistortion(p=0.5),
        A.OneOf(
            [
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.ElasticTransform(),
            ]
        ),
        A.OneOf(
            [
                A.CoarseDropout(
                    max_holes=1,
                    max_width=int(crop_size * 0.4),
                    max_height=int(crop_size * 0.6),
                    fill_value=0,
                    p=0.5,
                ),
                # A.ChannelShuffle(p=0.5),
                # A.ChannelDropout(p=0.5),
                A.Cutout(
                    p=0.5,
                    num_holes=2,
                    max_h_size=int(crop_size * 0.6),
                    max_w_size=int(crop_size * 0.4),
                    fill_value=0,
                ),
            ]
        ),
        A.Normalize(mean=[0] * fragment_depth, std=[1] * fragment_depth),
        # A.RandomCrop(crop_size, crop_size),
        ToTensorV2(transpose_mask=True),
    ]
    use_tta: bool = True
    tta_transforms = tta.Compose(
        [
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

    soft_train_compose = [
        A.RandomResizedCrop(crop_size, crop_size, scale=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0] * fragment_depth, std=[1] * fragment_depth),
        ToTensorV2(transpose_mask=True),
    ]

    # --- Infer Config
    is_tta: bool = True
    ink_threshold: float = 0.5
    checkpoints: list[str] = field(default_factory=lambda: [""])


second_stage_config = dict(
    exp_name="exp055_2_2nd_stage",
    crop_size=768,
    positive_sample_only=True,
    resume_train=True,
    batch_size=16,
    valid_batch_size=16,
    epoch=20,
    start_awp_epoch=20,
    start_freaze_model_epoch=1,
)


def is_skip_test() -> bool:
    import hashlib

    a_file = f"{DATA_DIR}/a/mask.png"
    with open(a_file, "rb") as f:
        hash_md5 = hashlib.md5(f.read()).hexdigest()
    return hash_md5 == "0b0fffdc0e88be226673846a143bb3e0"


def binarize(img: np.ndarray, thr: float = 0.5) -> np.ndarray:
    img = img - img.min()
    img = img / (img.max() + 1e-7)
    img = (img > thr).astype(np.float32)
    return img


def read_image_mask(cfg: CFG, fragment_id: int, fragment_z: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """画像とマスクを読み込む

    Args:
        cfg: 設定
        fragment_id: 画像のID

    Returns:
        (volume, mask)
    """
    crop_size = cfg.crop_size

    logger.info(f"fragment_z: {fragment_z}")

    # --- read volume
    volume = []
    for i in range(fragment_z[0], fragment_z[1]):
        image = cv2.imread(str(DATA_DIR / f"train/{fragment_id}/surface_volume/{i:02}.tif"), 0)
        assert image.ndim == 2
        pad0 = crop_size - image.shape[0] % crop_size
        pad1 = crop_size - image.shape[1] % crop_size
        image = np.pad(image, pad_width=[(0, pad0), (0, pad1)], constant_values=0)
        volume.append(image)
    # (h, w, depth)
    volume = np.stack(volume, axis=2)

    # --- read mask
    mask = cv2.imread(str(DATA_DIR / f"train/{fragment_id}/inklabels.png"), 0)
    mask = np.pad(mask, pad_width=[(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype(np.float32)
    mask /= 255.0

    return volume, mask


@dataclass(frozen=True)
class Data:
    volume: np.ndarray
    """(height, width, depth)"""
    fragment_id: int
    """0 ~ 65"""
    label: Optional[np.ndarray] = None
    """inklabels"""
    mask: Optional[np.ndarray] = None
    """mask"""
    ir: Optional[np.ndarray] = None
    "ir"


def read_data(mode: list[str], fragment_id: int, fragment_z: tuple[int, int]) -> Data:
    volume = []
    for i in range(fragment_z[0], fragment_z[1]):
        v_i = np.array(
            Image.open(DATA_DIR / f"{fragment_id}" / "surface_volume" / f"{i:02d}.tif"),
            dtype=np.uint16,
        )
        v_i = (v_i >> 8).astype(np.uint8)
        volume.append(v_i)
        logger.info(f"read_data: volume-{i:02d} {v_i.shape} {v_i.dtype}")
    volume = np.stack(volume, axis=-1)
    # height, width, depth = volume.shape

    mask = cv2.imread(f"{DATA_DIR}/{fragment_id}/mask.png", cv2.IMREAD_GRAYSCALE)
    mask = binarize(mask)

    if "train" in mode:
        ir = cv2.imread(f"{DATA_DIR}/{fragment_id}/ir.png", cv2.IMREAD_GRAYSCALE)
        ir = ir / 255
        label = cv2.imread(f"{DATA_DIR}/{fragment_id}/inklabels.png", cv2.IMREAD_GRAYSCALE)
        label = binarize(label)
        return Data(volume, fragment_id, label, mask, ir)
    else:
        return Data(volume, fragment_id, None, mask, None)


def read_data2(cfg: CFG, fragment_id: str | int) -> Data:
    """データを読み込む

    Args:
        cfg (CFG):
        fragment_id (str | int): 0 ~ 65, 2a, 2b

    Returns:
        data: Data
    """
    mode = cfg.mode
    fragment_z = cfg.fragment_z

    if fragment_id == "2a":
        y = 9456
        data = read_data(mode, 2, fragment_z)
        assert "train" in mode
        data = Data(
            volume=data.volume[:y, :, :],
            fragment_id=2,
            label=data.label[:y, :],
            mask=data.mask[:y, :],
            ir=data.ir[:y, :],
        )
        return data
    if fragment_id == "2b":
        y = 9456
        data = read_data(mode, 2, fragment_z)
        assert "train" in mode
        data = Data(
            volume=data.volume[y:, :, :],
            fragment_id=2,
            label=data.label[y:, :],
            mask=data.mask[y:, :],
            ir=data.ir[y:, :],
        )
        return data

    data = read_data(mode, fragment_id, fragment_z)
    return data


def get_train_valid_split(
    cfg: CFG, valid_id: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray,]:
    """訓練データと検証データを分割する

    Args:
        cfg: 設定
        valid_id: 検証データのID, {1: 1, 2: 2_1, 3: 2_2, 4: 2_3, 5: 3}

    Returns:
        (train_images, train_masks, valid_images, valid_masks, valid_xyxys)
    """
    if not (1 <= valid_id <= 5):
        raise ValueError(f"Invalid valid_id: {valid_id}")

    crop_size = cfg.crop_size
    stride = cfg.stride
    fragment_z = cfg.fragment_z
    is_second_stage = cfg.is_second_stage

    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    n_split_fragment = 2

    for fragment_id in range(1, 4):
        # image: (h, w, in_chans)
        # mask: (h, w)
        image, mask = read_image_mask(cfg=cfg, fragment_id=fragment_id, fragment_z=fragment_z)
        logger.info(f"fragment_id={fragment_id}, image.shape={image.shape}, mask.shape={mask.shape}")

        x1_list = list(range(0, mask.shape[1] - crop_size + 1, stride))
        y1_list = list(range(0, mask.shape[0] - crop_size + 1, stride))
        fragment_id_2_split_range = np.arange(0, mask.shape[0], mask.shape[0] // n_split_fragment)

        # if n_split == 3:
        # fragment_id_2_split_range = [0, 5002, 10004, 15006]
        # valid_id = 2 -> 0 ~ 5002
        # valid_id = 3 -> 5002 ~ 10004
        # valid_id = 4 -> 10004 ~ 15006

        # if n_split == 2:
        # fragment_id_2_split_range = [0, 7503, 15006]

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + crop_size
                x2 = x1 + crop_size

                contains_ink = mask[y1:y2, x1:x2].sum() > 0
                if is_second_stage and not contains_ink:
                    continue

                # y1の値で三等分するよりも、y2の値で三等分した方が、境界付近のリーク減りそう
                is_fold1 = fragment_id == 1 and valid_id == 1
                is_fold2 = (
                    fragment_id == 2
                    and valid_id == 2
                    and fragment_id_2_split_range[0] <= y2 < fragment_id_2_split_range[1]
                )
                is_fold3 = fragment_id == 2 and valid_id == 3 and fragment_id_2_split_range[1] <= y2
                # is_fold4 = fragment_id == 2 and valid_id == 4 and fragment_id_2_split_range[2] <= y2
                is_fold4 = fragment_id == 3 and valid_id == 5
                if is_fold1 or is_fold2 or is_fold3 or is_fold4:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
    valid_xyxys = np.array(valid_xyxys)
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


class VCDataset(Dataset):
    """
    Args:
        images: (N, H, W)
        labels:
        phase: {"train", "valid", "test"}
        crop_size:
        transform_fn:
    """

    def __init__(
        self,
        images: list[np.ndarray],
        labels: list[np.ndarray] | None,
        phase: str = "train",
        transform_fn: Callable | tuple[Callable, Callable] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.phase = phase
        if phase == "train":
            if not isinstance(transform_fn, tuple):
                raise ValueError("Expedted transform_fn to be tuple, but got not tuple.")
            self.transform_fn, self.soft_transform_fn = transform_fn
        else:
            self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Returns:
            train: (hard_image, hard_mask, soft_image, soft_mask)
            valid: (image, mask)
            test: (image)
        """
        image = self.images[idx]
        # phase == "test"
        if self.labels is None:
            if self.transform_fn is not None:
                augmented = self.transform_fn(image=image)
                image = augmented["image"]
                return image
            else:
                return torch.tensor(image)
        # phase == "train" or "valid"
        else:
            mask = self.labels[idx]
            if self.transform_fn is None:
                raise ValueError("transform_fn is None.")
            if isinstance(self.transform_fn, tuple):
                raise ValueError("Expected transform_fn to be not tuple.")

            # if self.phase == "train":
            #     hard_augmented = self.transform_fn(image=image, mask=mask)
            #     soft_augmented = self.soft_transform_fn(image=image, mask=mask)
            #     images ={"hard": hard_augmented["images"], "soft": soft_augmented["images"]}
            #     masks ={"hard": hard_augmented["mask"], "soft": soft_augmented["mask"]}
            #     return images, masks

            augmented = self.transform_fn(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            return image, mask


def get_loaders(
    cfg: CFG,
    train_images: list[np.ndarray],
    train_labels: list[np.ndarray],
    valid_images: list[np.ndarray],
    valid_labels: list[np.ndarray],
) -> tuple[DataLoader, DataLoader]:
    """

    Returns:
        (train_loader, valid_loader)

        train_loader: (hard_image, hard_mask, soft_image, soft_mask)
        valid_loader: (image, mask)
    """
    train_batch_size = cfg.batch_size
    valid_batch_size = cfg.valid_batch_size
    num_workers = cfg.num_workers
    fragment_depth = cfg.fragment_depth
    use_amp = cfg.use_amp

    train_dataset = VCDataset(
        images=train_images,
        labels=train_labels,
        phase="train",
        transform_fn=get_alb_transforms(cfg=cfg, phase="train"),
    )
    valid_dataset = VCDataset(
        images=valid_images,
        labels=valid_labels,
        phase="valid",
        transform_fn=get_alb_transforms(cfg=cfg, phase="valid"),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    train_loader = PrefetchLoader(train_loader, std=1, mean=0, channels=fragment_depth, fp16=use_amp)
    valid_loader = PrefetchLoader(valid_loader, std=1, mean=0, channels=fragment_depth, fp16=use_amp)
    return train_loader, valid_loader


# =================================================
# Model
# =================================================
class SmpUnetDecorder(nn.Module):
    def __init__(self, in_channel: int, out_channels: list[int], skip_channels: list[int]) -> None:
        super().__init__()
        self.center = nn.Identity()

        in_channels = [
            in_channel,
        ] + out_channels[:-1]
        # assert len(in_channels) == len(out_channels), f"{len(in_channels)} != {len(out_channels)}"
        # assert len(skip_channels) == len(out_channels), f"{len(skip_channels)} != {len(out_channels)}"
        # [layer4, layer3, layer2, layer1, layer0]
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=in_chan,
                    skip_channels=skip_chan,
                    out_channels=out_chan,
                    use_batchnorm=True,
                    attention_type=None,
                )
                for in_chan, skip_chan, out_chan in zip(in_channels, skip_channels, out_channels)
            ]
        )
        self.last_block = DecoderBlock(
            in_channels=out_channels[-1],
            skip_channels=0,
            out_channels=out_channels[-1] // 2,
            use_batchnorm=True,
            attention_type=None,
        )
        # print(self.blocks)
        # if len(self.blocks) != 5:
        #     raise ValueError(
        #         f"Expected len(self.blocks) == 5, but got {len(self.blocks)}"
        #         + f", in_channels: {in_channels}, out_channels: {out_channels}, skip_channels: {skip_channels}"
        #     )

    def forward(self, feature: torch.Tensor, skip: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature (torch.Tensor): (batch_size, in_channels, height, width)
            skip (list[torch.Tensor]): [(batch_size, skip_channels, height, width), ...]

        Returns:
            last (torch.Tensor): (batch_size, out_channels[-1], height, width)
            decode (list[torch.Tensor]): [(batch_size, out_channels[i], height, width), ...]
        """
        dense = self.center(feature)
        # [crop_size, crop_size/2, crop_size/4, crop_size/8, crop_size/16]
        decode = []
        for i, block in enumerate(self.blocks):
            dense = block(dense, skip[i])
            # print("i:", i, "dense:", dense.shape)
            decode.append(dense)

        last = self.last_block(decode[-1], skip=None)
        # print("decode[-1]:", decode[-1].shape)
        # print(last.shape)
        return last, decode


class VCNet(nn.Module):
    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        self.crop_depth = cfg.crop_depth
        self.pretrained = cfg.pretrained
        self.volume_depth = cfg.fragment_depth
        self.aux_params = cfg.aux_params

        self.output_type = ["inference", "loss"]

        conv_dim = 64
        encoder1_dims = [conv_dim, 64, 128, 256, 512]
        decoder1_dims = [256, 128, 64, 64]
        # encoder1_dims = [64, 64, 128, 256, 512]
        # decoder1_dims = [256, 128, 64, 64]

        self.encoder1 = resnet34d(pretrained=self.pretrained, in_chans=self.crop_depth)
        self.decoder1 = SmpUnetDecorder(
            in_channel=encoder1_dims[-1],
            skip_channels=encoder1_dims[:-1][::-1],
            out_channels=decoder1_dims,
        )

        # pool attention weights
        # [32, 32, 64, 128, 256, 512]
        self.weights1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for dim in encoder1_dims
            ]
        )
        # channel方向に集約
        self.logit1 = nn.Conv2d(in_channels=decoder1_dims[-1] // 2, out_channels=1, kernel_size=1)

        self.cls_head1 = ClassificationHead(in_channels=self.encoder1.feature_info[-1]["num_chs"], **self.aux_params)
        # print(self.encoder1.feature_info)

        # encoder2_dims = [64, 64, 128, 256, 512]
        # decoder2_dims = [256, 128, 64, 64]
        encoder2_dims = [64, 128, 256, 512]
        decoder2_dims = [128, 64, 32]
        self.encoder2 = resnet10t(pretrained=self.pretrained, in_chans=decoder1_dims[-1])
        self.decoder2 = SmpUnetDecorder(
            in_channel=encoder2_dims[-1],
            skip_channels=encoder2_dims[:-1][::-1],
            out_channels=decoder2_dims,
        )
        self.logit2 = nn.Conv2d(decoder2_dims[-1] // 2, 1, kernel_size=1)
        self.cls_head2 = ClassificationHead(in_channels=self.encoder2.feature_info[-1]["num_chs"], **self.aux_params)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """

        Args:
            batch (dict[str, torch.Tensor]): {
                "volume": (batch_size, 6, height, width),
                "label": (batch_size, height, width),
                "mask": (batch_size, height, width),
                "ir": (batch_size, height, width),
            }

        Returns:
            dict[str, torch.Tensor]: {
                "ink": (batch_size, height, width),
            }

        Ref:
            [1] https://www.kaggle.com/code/hengck23/lb0-68-one-fold-stacked-unet
            [2] https://www.kaggle.com/code/iafoss/hubmap-pytorch-fast-ai-starter
        """
        volume = batch["volume"]
        B, C, H, W = volume.shape
        vv = [volume[:, i : i + self.crop_depth] for i in range(0, self.volume_depth - self.crop_depth + 1, 2)]
        K = len(vv)
        # shape: (K, B, C, H, W)
        x = torch.cat(vv, dim=0)

        _x = rearrange(x, "(K B) c h w -> K B c h w", K=K, B=B, h=H, w=W)
        # print(K, B, C, H, W)
        # print(_x.shape)
        cls_logits1 = []
        for i in range(K):
            cls_features = self.encoder1.forward_features(_x[i])
            cls_logit1 = self.cls_head1(cls_features)
            cls_logits1.append(cls_logit1)
        # (K, B, 1)
        cls_logits1 = torch.stack(cls_logits1, dim=0).reshape(B, K)
        cls_logits1 = cls_logits1.mean(dim=1).reshape(B, 1)
        if cls_logits1.shape != (B, 1):
            raise ValueError(f"{cls_logits1.shape}")

        # ------------------------
        encoder = []

        e = self.encoder1
        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x)
        encoder.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x)
        encoder.append(x)
        x = e.layer2(x)
        encoder.append(x)
        x = e.layer3(x)
        encoder.append(x)
        x = e.layer4(x)
        encoder.append(x)

        # print("encoder", [f.shape for f in encoder])
        # print(self.weights1[0])
        # print(self.weights1[1])
        # print(self.weights1[2])

        for i in range(len(encoder)):
            e = encoder[i]
            # e = F.avg_pool2d(e, kernel_size=4, stride=4)
            f = self.weights1[i](e)
            _, c, h, w = e.shape
            f = rearrange(f, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)
            e = rearrange(e, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)
            w = F.softmax(f, dim=1)
            e = (w * e).sum(dim=1)
            # print(f"i={i}, e.shape={e.shape}")
            encoder[i] = e
        # pooled encoder = [crop_size/8, crop_size/16, crop_size/32, crop_size/64, crop_size/128]
        # where kernel_size=4, stride=4

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last1, decoder1 = self.decoder1(feature, skip)
        last1 = F.dropout(last1, p=0.5, training=self.training)
        # (B, 1, H/2, W/2)
        logits1 = self.logit1(last1)
        # print("last1", last1.shape)
        # print("logits1", logits1.shape)
        # logits1 = F.interpolate(logits1, size=(H, W), mode="bilinear", align_corners=False)

        # [decode_layer4, decode_layer3, decode_layer2, decode_layer1, decode_layer0]
        x = F.dropout(decoder1[-1], p=0.5, training=self.training)
        # print("x at decoder1[-1]", x.shape)

        # _x = rearrange(x, "(K B) c h w -> K B c h w", K=K, B=B, h=H, w=W)
        # cls_logits2 = []
        # for i in range(K):
        #     cls_features = self.encoder2.forward_features(_x[i])
        #     cls_logit2 = self.cls_head2(cls_features)
        #     cls_logits2.append(cls_logit2)
        # # (K, B, 1)
        # cls_logits2 = torch.stack(cls_logits2, dim=0).reshape(B, K)
        # cls_logits2 = cls_logits2.mean(dim=1)
        # assert cls_logits2.shape == (B, 1)

        cls_features = self.encoder2.forward_features(x)
        cls_logit2 = self.cls_head2(cls_features)
        if cls_logit2.shape != (B, 1):
            raise ValueError(f"{cls_logit2.shape}")

        # ------------------------
        encoder = []
        e = self.encoder2
        x = e.layer1(x)
        encoder.append(x)
        x = e.layer2(x)
        encoder.append(x)
        x = e.layer3(x)
        encoder.append(x)
        x = e.layer4(x)
        encoder.append(x)

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last2, decoder2 = self.decoder2(feature, skip)
        last2 = F.dropout(last2, p=0.5, training=self.training)
        # (B, 1, H/2, W/2)
        logits2 = self.logit2(last2)
        # logits2 = F.interpolate(logits2, size=(H, W), mode="bilinear", align_corners=False)
        ink = torch.sigmoid(logits2)
        # ink = torch.sigmoid(F.interpolate(logits2, size=(H, W), mode="bilinear", align_corners=False))

        output = {
            "ink": ink,
            "logit1": logits1,
            "logit2": logits2,
            "cls_logits1": cls_logit1,
            "cls_logits2": cls_logit2,
        }
        return output


# =================================================
# train funcions
# =================================================
def get_mask(cfg: CFG, fragment_id: int) -> np.ndarray:
    valid_mask_gt = cv2.imread(str(DATA_DIR / f"train/{fragment_id}/inklabels.png"), 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = cfg.crop_size - valid_mask_gt.shape[0] % cfg.crop_size
    pad1 = cfg.crop_size - valid_mask_gt.shape[1] % cfg.crop_size
    valid_mask_gt = np.pad(valid_mask_gt, ((0, pad0), (0, pad1)), constant_values=0)
    return valid_mask_gt


def training_fn(cfg: CFG) -> None:
    exp_name = cfg.exp_name
    device = cfg.device
    patience = cfg.patience
    n_fold = cfg.n_fold
    epoch = cfg.epoch
    random_state = cfg.random_state
    use_amp = cfg.use_amp
    start_awp_epoch = cfg.start_awp_epoch
    resume_train = cfg.resume_train
    resume_weights = cfg.resume_weights
    schedule_per_epoch = cfg.schedule_per_epoch
    is_second_stage = cfg.is_second_stage
    train_folds = cfg.train_fold

    for fold in range(1, n_fold + 1):
        if fold not in train_folds:
            continue
        seed_everything(seed=random_state)
        logger.info("\n" + "=" * 30 + f" Fold {fold} " + "=" * 30 + "\n")
        (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            valid_xyxys,
        ) = get_train_valid_split(cfg=cfg, valid_id=fold)

        logger.info(f"train_images.shape: {len(train_images)}")
        logger.info(f"train_labels.shape: {len(train_labels)}")
        logger.info(f"valid_images.shape: {len(valid_images)}")
        logger.info(f"valid_labels.shape: {len(valid_labels)}")

        # -- fragement2//3
        # fragment_id = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}[fold]
        # -- fragement2//2
        fragment_id = {1: 1, 2: 2, 3: 2, 4: 3}[fold]
        valid_mask_gt = get_mask(cfg=cfg, fragment_id=fragment_id)
        net = VCNet(cfg=cfg).to(device=device, non_blocking=True)
        if resume_train:
            net.load_state_dict(torch.load(resume_weights[fold - 1], map_location=device))

        train_loader, valid_loader = get_loaders(
            cfg=cfg,
            train_images=train_images,
            train_labels=train_labels,
            valid_images=valid_images,
            valid_labels=valid_labels,
        )
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            fold=str(fold),
            save_dir=OUTPUT_DIR / exp_name,
            logger_fn=logger.info,
        )
        criterion = get_loss(cfg=cfg)
        criterion_cls = nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(cfg=cfg, model=net)
        scheduler = get_scheduler(cfg=cfg, optimizer=optimizer, step_per_epoch=len(train_loader))
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=use_amp)

        best_score = 0
        use_awp = False
        for epoch in range(epoch):
            train_avg_loss = train_per_epoch(
                cfg=cfg,
                model=net,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                fold=fold,
                epoch=epoch,
                criterion_cls=criterion_cls,
                use_awp=use_awp,
                schedule_per_step=not schedule_per_epoch,
            )
            valid_assets = valid_per_epoch(
                cfg=cfg,
                model=net,
                valid_loader=valid_loader,
                criterion=criterion,
                fold=fold,
                epoch=epoch,
                valid_xyxys=valid_xyxys,
                valid_masks=valid_mask_gt,
            )
            valid_avg_loss = valid_assets["valid_loss_avg"]
            mask_preds = valid_assets["mask_preds"]
            valid_avg_bce = valid_assets["valid_bce_avg"]
            if schedule_per_epoch:
                scheduler.step()
            best_dice, best_th = calc_cv(mask_gt=valid_mask_gt, mask_pred=mask_preds)
            score = best_dice
            wandb.log(
                {
                    f"fold{fold}_train_avg_loss": train_avg_loss,
                    f"fold{fold}_valid_avg_loss": valid_avg_loss,
                    f"fold{fold}_valid_avg_bce": valid_avg_bce,
                    f"fold{fold}_best_valid_dice": best_dice,
                    f"fold{fold}_best_valid_th": best_th,
                }
            )

            logger.info(
                f"\n\n\tEpoch {epoch} - train_avg_loss: {train_avg_loss} valid_avg_loss: {valid_avg_loss}"
                + f" best dice: {best_dice} best th: {best_th}\n\n"
            )
            if epoch > start_awp_epoch:
                if not use_awp:
                    logger.info(f"Start using awp at epoch {epoch}")
                use_awp = True

            if score > best_score:
                best_score = score
                torch.save(
                    {"preds": mask_preds, "best_dice": best_dice, "best_th": best_th},
                    str(CP_DIR / exp_name / f"best_fold{fold}{'_second' if is_second_stage else ''}.pth"),
                )

            early_stopping(val_loss=score, model=net)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        early_stopping.save_checkpoint(val_loss=0, model=net, prefix="last-")

        # --- make best pred mask
        mask_preds = torch.load(CP_DIR / exp_name / f"best_fold{fold}.pth")["preds"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes[0].imshow(valid_mask_gt)
        axes[0].set_title("GT")
        axes[1].imshow(mask_preds)
        axes[1].set_title("Pred")
        axes[2].imshow((mask_preds >= best_th).astype(np.uint8))
        axes[2].set_title("Pred with threshold")
        fig.savefig(OUTPUT_DIR / exp_name / f"pred_mask_fold{fold}.png")

    # --- report OOF score
    best_dices = [torch.load(CP_DIR / exp_name / f"best_fold{fold}.pth")["best_dice"] for fold in range(1, n_fold + 1)]
    mean_dice = np.mean(best_dices)

    logger.info(f"best dices: {best_dices}")
    logger.info("OOF mean dice: {}".format(mean_dice))
    wandb.log({"OOF mean dice": mean_dice, "best dices": best_dices})

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Training has finished.")


def fbeta_numpy(targets: np.ndarray, preds: np.ndarray, beta: float = 0.5, smooth: float = 1e-5) -> np.float32:
    """Compute fbeta score
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288

    Args:
        targets (np.ndarray): (N, H, W, C)
        preds (np.ndarray): (N, H, W, C)
        beta (float): beta. Defaults to 0.5.
        smooth (float): smooth. Defaults to 1e-5.

    Returns:
        float: fbeta score
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()  # true positive
    cfp = preds[targets == 0].sum()  # false positive
    beta_square = beta**2

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    fbeta = (1 + beta_square) * c_precision * c_recall / (beta_square * c_precision + c_recall + smooth)
    return fbeta


def calc_fbeta(mask, mask_pred):
    """Calculate fbeta score

    Args:
        mask (np.ndarray): (H, W, C)
        mask_pred (np.ndarray): (H, W, C)
    Returns:
        best_dice (float): best dice score
        best_th (float): best threshold
    """
    mask = mask.astype(np.int16).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0.0
    best_dice = 0.0
    for th in np.arange(0.1, 0.95, 0.05):
        dice = fbeta_numpy(mask, (mask_pred > th).astype(np.int16), beta=0.5)
        logger.info(f"th: {th}, dice: {dice}")
        if dice > best_dice:
            best_dice = float(dice)
            best_th = float(th)
    logger.info(f"best_th: {best_th}, best_dice: {best_dice}")
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    """Calculate cv score

    Args:
        mask_gt (np.ndarray): (H, W, C)
        mask_pred (np.ndarray): (H, W, C)
    Returns:
        best_dice (float): best dice score
        best_th (float): best threshold
    """
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)
    return best_dice, best_th


# =================================================
# Inference functions
# =================================================
def mask_to_rle(mask: np.ndarray) -> str:
    """
    Args:
        mask (np.ndarray): (height, width)

    Returns:
        rle (str): run-length encoding

    Ref:
        [1] https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0
    """
    flattened_mask = mask.reshape(-1)
    s = np.array((flattened_mask[:-1] == 0) & (flattened_mask[1:] == 1))
    e = np.array((flattened_mask[:-1] == 1) & (flattened_mask[1:] == 0))
    s_index = np.where(s)[0] + 2
    e_index = np.where(e)[0] + 2
    length = e_index - s_index
    rle = " ".join(map(str, sum(zip(s_index, length), ())))
    return rle


def calc_metric(
    thr: float, ink: np.ndarray, label: np.ndarray, mask_sum: np.ndarray, beta=0.5
) -> tuple[float, float, float, float, float]:
    """
    Args:
        thr (float): threshold
        ink (np.ndarray): (height, width)
        label (np.ndarray): (height, width)
        mask_sum (np.ndarray): (height, width)
        beta (float, optional): beta. Defaults to 0.5.

    Returns:
        p_sum, precision, recall, fpr, dice, score
    """
    p = ink.reshape(-1)
    t = label.reshape(-1)
    p = (p > thr).astype(np.float32)
    t = (t > 0.5).astype(np.float32)

    tp = p * t
    precision = tp.sum() / (p.sum() + 1e-7)
    recall = tp.sum() / (t.sum() + 1e-7)

    fp = p * (1 - t)
    fpr = fp.sum() / (1 - t).sum()

    score = beta * beta / (1 + beta * beta) * 1 / recall + 1 / (1 + beta * beta) * 1 / precision
    score = 1 / score
    dice = 2 * tp.sum() / (p.sum() + t.sum() + 1e-7)
    p_sum = p.sum() / mask_sum
    return p_sum, precision, recall, fpr, dice, score


def calc_metrics(ink: np.ndarray, label: np.ndarray, mask: np.ndarray) -> dict:
    p = ink.reshape(-1)
    t = label.reshape(-1)
    pos = np.log(np.clip(p, 1e-7, 1))
    neg = np.log(np.clip(1 - p, 1e-7, 1))
    bce = -(pos * t + neg * (1 - t)).mean()

    mask_sum = mask.sum()
    metrics = {
        "thr": 0.0,
        "score": 0.0,
        "dice": 0.0,
        "fpr": 0.0,
        "bce": bce,
        "p_sum": 0,
        "recall": 0,
    }
    for thr in np.linspace(0, 1, 11):
        p_sum, precision, recall, fpr, dice, score = calc_metric(thr, ink, label, mask_sum)
        if score > metrics["score"]:
            metrics["thr"] = thr
            metrics["p_sum"] = p_sum
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["fpr"] = fpr
            metrics["dice"] = dice
            metrics["score"] = score
    return metrics


def metric_to_text(ink: np.ndarray, label: np.ndarray, mask: np.ndarray) -> str:
    text = []
    p = ink.reshape(-1)
    t = label.reshape(-1)
    pos = np.log(np.clip(p, 1e-7, 1))
    neg = np.log(np.clip(1 - p, 1e-7, 1))
    bce = -(pos * t + neg * (1 - t)).mean()
    text.append(f"bce: {bce:.4f}")

    mask_sum = mask.sum()
    text.append(f"p_sum thr prec recall fpr dice score")
    text.append("-" * 40)

    # 評価計算
    for thr in np.linspace(0, 1, 11):
        p_sum, precision, recall, fpr, dice, score = calc_metric(thr, ink, label, mask_sum)
        text.append(f"{p_sum:.4f} {thr:.2f} {precision:.4f} {recall:.4f} {fpr:.4f} {dice:.4f} {score:.4f}")

    return "\n".join(text)


def make_infer_mask(cfg: CFG) -> np.ndarray:
    s = cfg.crop_size
    f = cfg.crop_fade
    x = np.linspace(-1, 1, s)
    y = np.linspace(-1, 1, s)
    xx, yy = np.meshgrid(x, y)
    d = 1 - np.maximum(np.abs(xx), np.abs(yy))
    d1 = np.clip(d, 0, f / s * 2)
    d1 = d1 / d1.max()
    infer_mask = d1
    return infer_mask


def get_test_fragments_ids(cfg: CFG) -> list[str]:
    test_data_dir = DATA_DIR / "test"
    test_fragments_ids = list(test_data_dir.glob("*"))
    test_fragments_ids = sorted([p.name for p in test_fragments_ids])
    return test_fragments_ids


def infer_per_sample(cfg: CFG, nets: list[VCNet], data: Data) -> np.ndarray:
    for net in nets:
        net = net.to(cfg.device)
        net.eval()

    # get coordinate
    crop_size = cfg.crop_size
    stride = cfg.stride
    H, W, D = data.volume.shape

    # padding
    px, py = W % stride, H % stride
    if px != 0 or py != 0:
        px = stride - px
        py = stride - py
        pad_volume = np.pad(data.volume, ((0, py), (0, px), (0, 0)), constant_values=0)
    else:
        pad_volume = data.volume

    p_h, p_w, _ = pad_volume.shape
    x = np.arrange(0, p_w - crop_size + 1, stride)
    y = np.arrange(0, p_h - crop_size + 1, stride)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x, y], axis=-1).reshape(-1, 2)
    logger.info(f"H, W, P_H, P_W, len(xy): {H}, {W}, {p_h}, {p_w}, {len(xy)}")

    # inference
    infer_mask = make_infer_mask(cfg)
    prob = np.zeros((p_h, p_w), dtype=np.float32)
    count = np.zeros((p_h, p_w), dtype=np.float32)
    batch_iter = np.array_split(xy, len(xy) // 6)
    for t, xy0 in enumerate(batch_iter):
        volume = []
        for x0, y0 in xy0:
            v = pad_volume[y0 : y0 + crop_size, x0 : x0 + crop_size]
            volume.append(v)
        volume = np.stack(volume, axis=0)
        # shape: (batch, crop_size, crop_size, depth) -> (batch, depth, crop_size, crop_size)
        volume = np.ascontiguousarray(volume.transpose(0, 3, 1, 2))
        volume = volume / 255
        volume = torch.from_numpy(volume).float().to(cfg.device)
        batch = {"volume": volume}

        k = 0
        c = 0
        with torch.inference_mode():
            with torch.cuda.amp.autocast_mode(enabled=True):
                for net in nets:
                    if cfg.is_tta:
                        ink = tta_rotate(net, volume)
                        k += ink.data.cpu().numpy()
                        c += 1
                    else:
                        output = net(batch)
                        ink = output["ink"]
                        k += ink.data.cpu().numpy()
                        c += 1
        k = k / c

        batch_size = len(k)
        for b in range(batch_size):
            x0, y0 = xy0[b]
            prob[y0 : y0 + crop_size, x0 : x0 + crop_size] += k[b, 0] * infer_mask
            count[y0 : y0 + crop_size, x0 : x0 + crop_size] += infer_mask
        logger.info(f"t: {t} / {len(batch_iter)}")
    prob = prob / (count + 1e-7)
    prob = prob[:H, :W]
    return prob


def infer(cfg: CFG, fragement_ids: list[str], checkpoints: list[str | Path]) -> pd.DataFrame:
    nets = []
    for i, f in enumerate(checkpoints):
        logger.info(f"load {f}")
        net = VCNet(cfg=cfg)
        state = torch.load(f, map_location=lambda storage, loc: storage)
        print(net.load_state_dict(state["state_dict"], strict=True))
        nets.append(net)

    logger.info(f"stride: {cfg.stride}, crop_size: {cfg.crop_size}")
    sub = defaultdict(list)
    for t, fragment_id in enumerate(fragement_ids):
        logger.info(f"t: {t} / {len(fragement_ids)} -> fragment_id: {fragment_id}")
        data = read_data2(cfg, fragment_id)
        logger.info(f"volume.shape {data.volume.shape}, mask.shape {data.mask.shape}")

        ink_prob = infer_per_sample(cfg, nets, data)
        ink_prob = data.mask * ink_prob
        ink_pred = (ink_prob > cfg.ink_threshold).astype(np.uint8)

        sub["id"].append(fragment_id)
        sub["ink"].append(mask_to_rle(ink_pred))

        prob8 = (ink_prob * 255).astype(np.uint8)
        plt.figure(t)
        plt.imshow(prob8, cmap="gray")

        if "train" in cfg.mode:
            metrics = calc_metrics(ink_pred, data.label, data.mask)
            wandb.log(metrics)
            text = metric_to_text(ink_pred, data.label, data.mask)
            logger.info(text)

    sub_df = pd.DataFrame.from_dict(sub)
    return sub_df


def main() -> None:
    cfg = CFG()
    if cfg.is_second_stage:
        logger.info("########## This is second stage ##########")
        cfg = CFG(**second_stage_config)
    logger.info(f"{cfg.__dict__}")
    seed_everything(seed=cfg.random_state)
    (OUTPUT_DIR / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    logger.info(f"Start Time = {start_time.strftime('%Y-%m-%d-%H-%M-%S')}")
    if "train" in cfg.mode:
        wandb.init(
            project="vesuvius_challenge",
            config=asdict(cfg),
            name=f"{cfg.exp_name}_{start_time}",
        )
        training_fn(cfg)
        wandb.finish()
        train_duration = datetime.now() - start_time
        logger.info(f"Train Duration = {train_duration}")

    elif "test" in cfg.mode:
        test_fragments_ids = get_test_fragments_ids(cfg)
        if is_skip_test():
            sub_df = pd.DataFrame({"Id": test_fragments_ids, "Predicted": ["1 2", "1 2"]})
        else:
            sub_df = infer(cfg, test_fragments_ids, cfg.checkpoints)
            print(sub_df)
        sub_df.to_csv("submission.csv", index=False)
        logger.info("make sub finish")


if __name__ == "__main__":
    main()
