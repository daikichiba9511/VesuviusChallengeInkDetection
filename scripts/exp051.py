"""exp044

- copy from exp042
- 2.5D segmentation

DIFF:

- effb4

Reference:
[1]
https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code
[2]
https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-inference
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import pickle
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import ttach as tta
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.metrics import roc_auc_score
from timm.data.loader import PrefetchLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from warmup_scheduler import GradualWarmupScheduler

import wandb
from src.augmentations import cutmix
from src.losses import soft_dice_loss

dbg = logger.debug

# ssl._create_default_https_context = ssl._create_unverified_context
# torchvision.disable_beta_transforms_warning()
warnings.simplefilter("ignore")


IS_TRAIN = not Path("/kaggle/working").exists()
MAKE_SUB: bool = False
SKIP_TRAIN = False

logger.info(
    f"Meta Config: IS_TRAIN={IS_TRAIN},"
    + f"MAKE_SUB={MAKE_SUB},"
    + f"SKIP_TRAIN={SKIP_TRAIN}"
)


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

THR = 0.5


def to_pickle(obj: Any, filename: Path) -> None:
    with filename.open("wb") as fp:
        pickle.dump(obj, fp)


def from_pickle(filename: Path) -> Any:
    with filename.open("wb") as fp:
        obj = pickle.load(fp)
    return obj


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(False)


# ==============================================================
# Config
# ==============================================================
@dataclass(frozen=True)
class CFG:
    # ================= Global cfg =====================
    exp_name = (
        "exp051_fold5_Unet++_seresnext50_gradualwarm_cutmix_mixup_tile224_slide74"
    )
    random_state = 42
    tile_size: int = 224
    image_size = (tile_size, tile_size)
    stride: int = tile_size // 3
    num_workers = mp.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ================= Train cfg =====================
    n_fold = 5  # [1, 2_1, 2_2, 2_3, 3]
    epoch = 15
    batch_size = 8 * 12
    use_amp: bool = True
    patience = 10

    optimizer = "AdamW"
    # optimizer = "RAdam"

    # optimizer params group lr
    warmup_factor = 10
    # encoder_lr = 3e-6 / warmup_factor
    # decoder_lr = 3e-5 / warmup_factor
    encoder_lr = 5e-4 / warmup_factor
    decoder_lr = 5e-3 / warmup_factor
    # encoder_lr = 1e-3 / warmup_factor
    # decoder_lr = 1e-2 / warmup_factor
    weight_decay = 5e-5

    scheduler = "GradualWarmupScheduler"
    # scheduler = "OneCycleLR"
    # scheduler = "TwoCyclicLR"
    # scheduler = "CosineAnnealingWarmRestarts"
    # scheduler = "CosineLRScheduler"  # timm

    # CosineAnnealingWarmRestartsの設定
    # min_lr = 1e-6
    # """learning rateの下限"""
    # T_0 = 10
    # """CosineAnnealingWarmRestartsの周期"""
    # T_mult = 1
    # """CosineAnnealingWarmRestartsの周期の増加率"""

    # CosineLRSchedulerの設定
    # t_initial = 2
    # """CosineLRSchedulerの周期"""
    # min_lr = 1e-6
    # """learning rateの下限"""
    # warmup_t = 20
    # """warmupの期間"""
    # warmup_lr_init = 1e-6
    # """warmupの初期値"""
    # warmup_prefix = True

    # OneCycleLR
    # max_lr = 1e-4

    # GradualWarmupSchedulerの設定
    T_max = epoch // 3

    max_grad_norm = 1000.0

    # AWP params
    start_awp = 10
    start_epoch = 10
    adv_lr = 1e-7
    # adv_lr = 1e-6
    # adv_lr = 1e-5
    adv_eps = 3
    adv_step = 1

    # when to start soft augmentation
    # if epoch < start_soft_aug_epoch, use hard augmentation
    start_soft_aug_epoch: int = epoch

    # num step of grad accumulation
    grad_accum = 4

    # ================= Loss cfg =====================
    # loss = "BCEWithLogitsLoss"
    # loss = "BCETverskyLoss"
    # loss = "BCEDiceLoss"
    # loss = "BCEFocalLovaszLoss"
    loss = "BCEFocalDiceLoss"

    # loss weights
    weight_bce = 0.5
    weight_focal = 0.1
    # weight_cls = 0.05
    # weight_cls = 0.01
    # weight_cls = 0.1
    weight_cls = 0.2

    # ================= Model =====================
    arch: str = "UnetPlusPlus"
    # arch: str = "Unet"
    encoder_name: str = "se_resnext50_32x4d"
    # encoder_name: str = "timm-efficientnet-b1"
    # encoder_name: str = "timm-efficientnet-b7"
    # encoder_name: str = "timm-efficientnet-b4"

    # encoder_name: str = "tu-efficientnetv2_l"
    # encoder_name: str = "tu-tf_efficientnetv2_m_in21ft1k"

    in_chans: int = 7
    weights = "imagenet"
    # weights = "advprop"
    aux_params = {
        "classes": 1,
        "pooling": "avg",
        "dropout": 0.8,
    }

    # ================= Data cfg =====================
    mixup = True
    mixup_prob = 0.5
    mixup_alpha = 0.1

    cutmix = True
    cutmix_prob = 0.5
    cutmix_alpha = 0.1

    train_compose = [
        # A.Resize(image_size[0], image_size[1]),
        A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        # A.CLAHE(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_width=int(image_size[1] * 0.3),
            max_height=int(image_size[0] * 0.3),
            fill_value=0,
            p=0.5,
        ),
        A.Cutout(p=0.5),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    soft_train_compose = [
        A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    # ================= Test cfg =====================
    use_tta = True
    # tta_transforms = tta.aliases.d4_transform()
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            # tta.Scale(scales=[1.0, 1.5, 2.0, 4.0])
        ]
    )


# ===============================================================
# utils
# ===============================================================
def rle(image: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        image: 画像, (H, W, 3)
        threshold: 閾値
    """
    logger.info(f"image.shape: {image.shape}, threshold: {threshold}")
    flat_image = image.flatten()
    dbg(f"flat_image: {flat_image.shape}")
    flat_image = np.where((flat_image > threshold).copy(), 1, 0).astype(np.uint8)

    starts = np.asarray((flat_image[:-1] == 0) & (flat_image[1:] == 1))
    ends = np.asarray((flat_image[:-1] == 1) & (flat_image[1:] == 0))
    dbg(f"starts: {starts.shape}, ends: {ends.shape}")

    starts_idx = np.where(starts)[0] + 2
    ends_idx = np.where(ends)[0] + 2
    dbg(f"starts: {starts_idx.shape}, ends: {ends_idx.shape}")

    length = ends_idx - starts_idx
    return starts_idx, length


def fast_rle(img) -> str:
    """
    Args:
        img: 画像, (H, W, 3), 1 - mask, 0 - background
    Returns:
        rle: run-length as string formated
    Reference:
    [1]
    https://www.kaggle.com/stainsby/fast-tested-rle
    [2]
    https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-inference
    """
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def make_tile_array(
    valid_preds: list[np.ndarray],
    h_count: int,
    w_count: int,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """
    Args:
        image_size: (H, W)
    """
    stack_pred = np.vstack(valid_preds).reshape(-1, image_size[0], image_size[1])
    tile_array = [
        stack_pred[h_i * w_count : (h_i + 1) * w_count].astype(np.float32)
        for h_i in range(h_count)
    ]
    return tile_array


def concat_tile(image_list_2d: list[np.ndarray]) -> np.ndarray:
    """画像を格子状に並べる

    Args:
        image_list_2d: 画像を格子状に並べた二次元配列。(H, W, (h, w, 3))

    Reference:
    [1]
    https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
    """
    return cv2.vconcat([cv2.hconcat(image_list_h) for image_list_h in image_list_2d])


# ===============================================================
# Data
# ===============================================================
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


def get_surface_volume_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image1 = np.stack(
        [
            cv2.imread(str(DATA_DIR / f"train/1/surface_volume/{i:02}.tif"), 0)
            for i in tqdm(range(65))
        ]
    )

    image2 = np.stack(
        [
            cv2.imread(str(DATA_DIR / f"train/2/surface_volume/{i:02}.tif"), 0)
            for i in tqdm(range(65))
        ]
    )

    image3 = np.stack(
        [
            cv2.imread(str(DATA_DIR / f"train/3/surface_volume/{i:02}.tif"), 0)
            for i in tqdm(range(65))
        ]
    )
    return image1, image2, image3


def get_inklabels_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        (image1_label, image2_label, image3_label)
    """
    image1_label = cv2.imread(str(DATA_DIR / "train/1/inklabels.png"), 0)
    image2_label = cv2.imread(str(DATA_DIR / "train/2/inklabels.png"), 0)
    image3_label = cv2.imread(str(DATA_DIR / "train/3/inklabels.png"), 0)
    return image1_label, image2_label, image3_label


def get_mask_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        (image1_mask, image2_mask, image3_mask)
    """
    image1_mask = cv2.imread(str(DATA_DIR / "train/1/mask.png"), 0)
    image2_mask = cv2.imread(str(DATA_DIR / "train/2/mask.png"), 0)
    image3_mask = cv2.imread(str(DATA_DIR / "train/3/mask.png"), 0)
    return image1_mask, image2_mask, image3_mask


def read_image_mask(cfg: CFG, fragment_id: int) -> tuple[np.ndarray, np.ndarray]:
    """画像とマスクを読み込む

    Args:
        cfg: 設定
        fragment_id: 画像のID

    Returns:
        (image, mask)
    """
    images = []

    mid = 65 // 2
    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2 + 1
    idxs = range(start, end)
    for i in idxs:
        image = cv2.imread(
            str(DATA_DIR / f"train/{fragment_id}/surface_volume/{i:02}.tif"), 0
        )

        assert image.ndim == 2

        pad0 = cfg.tile_size - image.shape[0] % cfg.tile_size
        pad1 = cfg.tile_size - image.shape[1] % cfg.tile_size

        image = np.pad(image, pad_width=[(0, pad0), (0, pad1)], constant_values=0)
        images.append(image)

    # (h, w, in_chans)
    images = np.stack(images, axis=2)
    dbg(f"images.shape = {images.shape}")
    mask = cv2.imread(str(DATA_DIR / f"train/{fragment_id}/inklabels.png"), 0)
    mask = np.pad(mask, pad_width=[(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype(np.float32)
    mask /= 255.0
    return images, mask


def get_train_valid_split(
    cfg: CFG, valid_id: int
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[list[int]],
]:
    """訓練データと検証データを分割する

    Args:
        cfg: 設定
        valid_id: 検証データのID, {1: 1, 2: 2_1, 3: 2_2, 4: 2_3, 5: 3}

    Returns:
        (train_images, train_masks, valid_images, valid_masks, valid_xyxys)
    """
    if not (1 <= valid_id <= 5):
        raise ValueError(f"Invalid valid_id: {valid_id}")

    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):
        # image: (h, w, in_chans)
        # mask: (h, w)
        image, mask = read_image_mask(cfg=cfg, fragment_id=fragment_id)
        dbg(
            f"fragment_id={fragment_id}, image.shape={image.shape}, mask.shape={mask.shape}"
        )

        x1_list = list(range(0, mask.shape[1] - cfg.tile_size + 1, cfg.stride))
        y1_list = list(range(0, mask.shape[0] - cfg.tile_size + 1, cfg.stride))
        fragment_id_2_split_range = np.arange(0, mask.shape[0], mask.shape[0] // 3)
        # fragment_id_2_split_range = [0, 5002, 10004, 15006]
        # valid_id = 2 -> 0 ~ 5002
        # valid_id = 3 -> 5002 ~ 10004
        # valid_id = 4 -> 10004 ~ 15006

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size
                # y1の値で三等分するよりも、y2の値で三等分した方が、境界付近のリーク減りそう
                if (
                    (fragment_id == 1 and valid_id == 1)  # fold1
                    or (
                        fragment_id == 2
                        and valid_id == 2
                        and fragment_id_2_split_range[0]
                        <= y2
                        < fragment_id_2_split_range[1]
                    )  # fold2
                    or (
                        fragment_id == 2
                        and valid_id == 3
                        and fragment_id_2_split_range[1]
                        <= y2
                        < fragment_id_2_split_range[2]
                    )  # fold3
                    or (
                        fragment_id == 2
                        and valid_id == 4
                        and fragment_id_2_split_range[2]
                        <= y2
                        < fragment_id_2_split_range[3]
                    )  # fold4
                    or (fragment_id == 3 and valid_id == 5)  # fold5
                ):
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_alb_transforms(phase: str, cfg: CFG) -> A.Compose | tuple[A.Compose, A.Compose]:
    """
    Args:
        phase: {"train", "valid", "test"}
        cfg: 設定
    """
    if phase == "train":
        return A.Compose(cfg.train_compose), A.Compose(cfg.soft_train_compose)
    elif phase == "valid":
        return A.Compose(
            [
                A.Resize(cfg.image_size[0], cfg.image_size[1]),
                A.Normalize(mean=[0] * cfg.in_chans, std=[1] * cfg.in_chans),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif phase == "test":
        return A.Compose(
            [
                A.Resize(cfg.image_size[0], cfg.image_size[1]),
                A.Normalize(mean=[0] * cfg.in_chans, std=[1] * cfg.in_chans),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        raise ValueError(f"Invalid phase: {phase}")


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
        cfg: CFG,
        images: list[np.ndarray],
        labels: list[np.ndarray] | None,
        phase: str = "train",
        transform_fn: Callable | tuple[Callable, Callable] | None = None,
    ) -> None:
        self.cfg = cfg
        self.images = images
        self.labels = labels
        self.phase = phase
        if phase == "train":
            if not isinstance(transform_fn, tuple):
                raise ValueError(
                    "Expedted transform_fn to be tuple, but got not tuple."
                )
            self.transform_fn, self.soft_transform_fn = transform_fn
        else:
            self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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

            if self.phase == "train":
                hard_augmented = self.transform_fn(image=image, mask=mask)
                soft_augmented = self.soft_transform_fn(image=image, mask=mask)

                hard_image = hard_augmented["image"]
                hard_mask = hard_augmented["mask"]

                soft_image = soft_augmented["image"]
                soft_mask = soft_augmented["mask"]

                return hard_image, hard_mask, soft_image, soft_mask

            else:
                augmented = self.transform_fn(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                return image, mask


# =======================================================================
# Model
# =======================================================================
class VCNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        arch: str = "Unet",
        in_chans: int = 65,
        encoder_name: str = "resnet18",
        weights: str | None = "imagenet",
        aux_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_chans,
            classes=num_classes,
            activation=None,
            aux_params=aux_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W)
        Returns:
            masks: (N, 1, H, W)
            labels: (N, 1)
        """
        masks, labels = self.model(x)
        return {
            "pred_mask_logits": masks,
            "pred_label_logits": labels,
        }


def build_model(cfg: CFG) -> VCNet:
    logger.info(f"Build model: {cfg.arch} with {cfg.encoder_name} encoder")
    model = VCNet(
        num_classes=1,
        arch=cfg.arch,
        encoder_name=cfg.encoder_name,
        in_chans=cfg.in_chans,
        weights=None,
    )
    return model


class EnsembleModel:
    def __init__(self, cfg: CFG, use_tta: bool = False) -> None:
        self.cfg = cfg
        self.use_tta = use_tta
        self.models = []

    def add_model(self, model: VCNet) -> None:
        self.models.append(model)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [torch.sigmoid(model(x)).to("cpu") for model in self.models]
        avg_preds = torch.mean(torch.stack(outputs), axis=0)
        return avg_preds


def build_ensemble_model(cfg: CFG) -> EnsembleModel:
    ensemble_model = EnsembleModel(cfg=cfg, use_tta=cfg.use_tta)
    model_dir = CP_DIR / cfg.exp_name if IS_TRAIN else CP_DIR
    for fold in range(1, cfg.n_fold + 1):
        _model = build_model(cfg)
        _model = _model.to(cfg.device, non_blocking=True)
        model_path = model_dir / f"checkpoint_{fold}.pth"
        logger.info(f"Load model from {model_path}")
        state = torch.load(model_path)
        _model.load_state_dict(state)
        _model.eval()
        ensemble_model.add_model(_model)
    return ensemble_model


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        logger_fn: Callable = print,
        save_dir: Path = Path("./output"),
        fold: str = "0",
        save_prefix: str = "",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger_fn = logger_fn
        self.fold = fold
        self.save_prefix = save_prefix
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model)

        if score < self.best_score + self.delta:
            self.counter += 1
            self.logger_fn(
                f"EarlyStopping Counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.logger_fn(
                f"Detected Increasing Score: best score {self.best_score} --> {score}"
            )
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model)
            self.counter = 0

    def save_checkpoint(
        self, val_loss: float, model: nn.Module, prefix: str = ""
    ) -> None:
        """Save model when validation loss decrease."""
        if self.verbose:
            self.logger_fn(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss})"
            )

        state_dict = model.state_dict()
        save_prefix = prefix if prefix != "" else self.save_prefix
        save_path = self.save_dir / f"{save_prefix}checkpoint_{self.fold}.pth"
        torch.save(state_dict, save_path)
        self.val_loss_min = val_loss


class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}, Row values {self.rows}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.rows: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.rows.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {"name": self.name, "avg": self.avg, "row_values": self.rows}


def split_cv(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[dict[str, list[np.ndarray]]]:
    image1, image2, image3 = images
    image1_label, image2_label, image3_label = labels
    image1_mask, image2_mask, image3_mask = masks

    data_fold = []
    data_fold.append(
        {
            "train_image": [image1, image2],
            "train_label": [image1_label, image2_label],
            "valid_image": [image3],
            "valid_label": [image3_label],
            "valid_mask": [image3_mask],
        }
    )
    data_fold.append(
        {
            "train_image": [image1, image3],
            "train_label": [image1_label, image3_label],
            "valid_image": [image2],
            "valid_label": [image2_label],
            "valid_mask": [image2_mask],
        }
    )
    data_fold.append(
        {
            "train_image": [image2, image3],
            "train_label": [image2_label, image3_label],
            "valid_image": [image1],
            "valid_label": [image1_label],
            "valid_mask": [image1_mask],
        }
    )
    return data_fold


def dice_coef_torch(
    y_pred: torch.Tensor, y_true: torch.Tensor, beta: float = 0.5, smooth: float = 1e-5
) -> torch.Tensor:
    """Compute dice coefficient

    Args:
        y_pred (torch.Tensor): (N, 1, H, W)
        y_true (torch.Tensor): (N, 1, H, W)
        beta (float): beta. Defaults to 0.5.
        smooth (float): smooth. Defaults to 1e-5.

    Returns:
        float: dice coefficient
    """

    # flatten label and prediction tensors
    preds = y_pred.view(-1).float()
    targets = y_true.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_square = beta**2
    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (
        (1 + beta_square)
        * c_precision
        * c_recall
        / (beta_square * c_precision + c_recall + smooth)
    )
    return dice


def calc_dice(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate dice score

    Args:
        y_pred (torch.Tensor): (N, 1, H, W)
        y_true (torch.Tensor): (N, 1, H, W)
    Returns:
        dice (float): dice score
    """
    # dice = smp.losses.DiceLoss(mode="binary")(y_pred=y_pred, y_true=y_true).item()
    dice = dice_coef_torch(y_pred=y_pred, y_true=y_true)
    return dice


def fbeta_numpy(
    targets: np.ndarray, preds: np.ndarray, beta: float = 0.5, smooth: float = 1e-5
) -> np.float32:
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
    fbeta = (
        (1 + beta_square)
        * c_precision
        * c_recall
        / (beta_square * c_precision + c_recall + smooth)
    )
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


def plot_dataset(
    cfg: CFG, train_images: list[np.ndarray], train_labels: list[np.ndarray]
) -> None:
    """Plot dataset

    Args:
        train_images (np.ndarray): (N, H, W, C)
        train_labels (np.ndarray): (N, H, W)
    """
    transform = A.Compose(
        [
            t
            for t in get_alb_transforms(cfg=cfg, phase="train")
            if not isinstance(t, (A.Normalize, ToTensorV2))
        ]
    )
    dataset = VCDataset(cfg=cfg, images=train_images, labels=train_labels)

    plot_count = 0
    for i in range(1000):
        image, mask = dataset[i]
        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        if mask.sum() == 0:
            continue

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(image[..., 0], cmap="gray")
        ax[0].set_title("Original Image")
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Original Mask")
        ax[2].imshow(aug_image[..., 0], cmap="gray")
        ax[2].set_title("Augmented Image")
        ax[3].imshow(aug_mask, cmap="gray")
        ax[3].set_title("Augmented Mask")
        fig.savefig(fname=str(OUTPUT_DIR / cfg.exp_name / f"dataset_{i}.png"))

        plot_count += 1
        if plot_count >= 10:
            break


class AWP:
    """Adversarial Weight Perturbation

    Args:
        model (torch.nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        criterion (Callable): loss function
        adv_param (str): parameter name to be perturbed. Defaults to "weight".
        adv_lr (float): learning rate. Defaults to 0.2.
        adv_eps (int): epsilon. Defaults to 1.
        start_epoch (int): start epoch. Defaults to 0.
        adv_step (int): adversarial step. Defaults to 1.
        scaler (torch.cuda.amp.GradScaler): scaler. Defaults to None.

    Examples:
    >>> model = Model()
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    >>> batch_size = 16
    >>> epochs = 10
    >>> num_train_train_steps = int(len(train_images) / batch_size * epochs)
    >>> awp = AWP(
    ...    model=model,
    ...    optimizer=optimizer,
    ...    adv_lr=1e-5,
    ...    adv_eps=3,
    ...    start_epoch=num_train_steps // epochs,
    ...    scaler=None,
    ... )
    >>> awp.attack_backward(image, mask_label, epoch)

    References:
    1.
    https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    2.
    https://speakerdeck.com/masakiaota/kaggledeshi-yong-sarerudi-dui-xue-xi-fang-fa-awpnolun-wen-jie-shuo-toshi-zhuang-jie-shuo-adversarial-weight-perturbation-helps-robust-generalization
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        criterion: Callable,
        adv_param: str = "weight",
        adv_lr: float = 0.2,
        adv_eps: int = 1,
        start_epoch: int = 0,
        adv_step: int = 1,
        scaler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.criterion = criterion

    def attack_backward(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> None:
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None
        self._save()
        for i in range(self.adv_step):
            # modelを近傍の悪い方へ改変
            self._attack_step()
            with autocast(device_type="cuda", enabled=self.scaler is not None):
                logits = self.model(x)["pred_mask_logits"]
                adv_loss = self.criterion(logits, y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

        # awpする前のモデルに戻す
        self._restore()

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数でパラメータの勾配を取得できるようにしておく必要がある
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def make_cls_label(mask: torch.Tensor) -> torch.Tensor:
    """make classification label from mask

    Args:
        mask (torch.Tensor): mask

    Returns:
        torch.Tensor: classification label
    """

    # (labels.view(b, -1).sum(-1) > 0).float().view(b, 1)
    batch_size = len(mask)
    # shape: (N, 1)
    target_cls = (mask.view(batch_size, -1).sum(-1) > 0).float().view(batch_size, 1)
    return target_cls


# ==========================================================
# training function
# ==========================================================
def train_per_epoch(
    cfg,
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    scaler,
    fold: int,
    optimizer,
    scheduler,
    epoch: int,
    criterion_cls,
    use_awp: bool = False,
) -> float:
    awp = AWP(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        adv_lr=cfg.adv_lr,
        adv_eps=cfg.adv_eps,
        start_epoch=cfg.start_epoch,
        adv_step=cfg.adv_step,
    )
    running_loss = AverageMeter(name="train_loss")
    # train_losses = []
    with tqdm(
        enumerate(train_loader), total=len(train_loader), dynamic_ncols=True
    ) as pbar:
        for step, (image, target, soft_image, soft_mask) in pbar:
            model.train()

            if cfg.mixup and np.random.rand() <= cfg.mixup_prob:
                image, target, _, _ = mixup(image, target, alpha=cfg.mixup_alpha)

            if cfg.cutmix and np.random.rand() <= cfg.cutmix_prob:
                image, target, _, _ = cutmix(image, target, alpha=cfg.cutmix_alpha)

            # target: (N, H, W)
            if epoch < cfg.start_soft_aug_epoch:
                image = image.to(cfg.device, non_blocking=True)
                target = target.to(cfg.device, non_blocking=True)
            else:
                image = soft_image.to(cfg.device, non_blocking=True)
                target = soft_mask.to(cfg.device, non_blocking=True)

            # image = image.to(cfg.device, non_blocking=True)
            # target = target.to(cfg.device, non_blocking=True)
            batch_size = target.size(0)
            target_cls = make_cls_label(target)

            with autocast(device_type="cuda", enabled=cfg.use_amp):
                outputs = model(image)
                # (N, 1, H, W)
                pred_mask = outputs["pred_mask_logits"]
                # (N, 1)
                pred_label = outputs["pred_label_logits"]

                assert len(pred_mask.shape) == 4, f"Got {pred_mask.shape}"
                assert pred_mask.shape[1] == 1

                loss_mask = criterion(pred_mask.squeeze(1), target)
                loss_cls = criterion_cls(pred_label, target_cls)
                loss = loss_mask + (cfg.weight_cls * loss_cls)
                loss /= cfg.grad_accum

            running_loss.update(value=loss.item(), n=batch_size)

            # multiple lossの時は別々に勾配計算？
            # Ref:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
            scaler.scale(loss).backward()

            # scaler.scale(loss_mask).backward(retain_graph=True)
            # scaler.scale(loss_cls).backward()

            if use_awp:
                awp.attack_backward(image, target, step)

            if (step + 1) % cfg.grad_accum == 0:
                # unscale -> clip
                # Ref
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                # scaler.unscale_(optimizer)
                # clip gradient of parameters
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(
                    {
                        "fold": f"{fold}",
                        "epoch": f"{epoch}",
                        "loss": f"{loss.item():.4f}",
                    }
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        f"fold{fold}_train_loss": loss.item(),
                        "learning_rate": learning_rate,
                    }
                )
    return running_loss.avg


def valid_per_epoch(
    cfg: CFG,
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: Callable,
    fold: int,
    epoch: int,
    valid_xyxys: np.ndarray,
    valid_masks: torch.Tensor,
) -> tuple[float, np.ndarray]:
    mask_preds = np.zeros(valid_masks.shape)
    mask_count = np.zeros(valid_masks.shape)
    model.eval()
    valid_losses = AverageMeter(name="valid_loss")

    if cfg.use_tta:
        tta_model = tta.SegmentationTTAWrapper(
            model,
            cfg.tta_transforms,
            merge_mode="mean",
            output_mask_key="pred_mask_logits",
        )
    else:
        tta_model = model

    for step, (image, target) in tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        smoothing=0,
        dynamic_ncols=True,
        desc="Valid Per Epoch",
    ):
        image = image.to(cfg.device, non_blocking=True)
        target_cls = make_cls_label(target)
        target_cls = target_cls.to(cfg.device, non_blocking=True)
        target = target.to(cfg.device, non_blocking=True)
        batch_size = target.size(0)

        with torch.inference_mode():
            # segm_logits: (N, 1, H, W)
            y_preds = tta_model(image)["pred_mask_logits"]
            loss_mask = criterion(y_preds, target)

            # cls: (N, 1)
            pred = model(image)
            pred_logtis = pred["pred_label_logits"]
            loss_cls = nn.BCEWithLogitsLoss()(pred_logtis, target_cls)
            accs = ((pred_logtis > 0.5) == target_cls).sum().item() / batch_size
            loss = loss_mask + (cfg.weight_cls * loss_cls)

        valid_losses.update(value=loss.item(), n=batch_size)
        wandb.log(
            {
                f"fold{fold}_valid_loss": loss.item(),
                f"fold{fold}_valid_cls_loss": loss_cls.item(),
                f"fold{fold}_valid_acc": accs,
            }
        )

        # make a whole image prediction
        y_preds = torch.sigmoid(y_preds).to("cpu").detach().numpy()
        start_idx = step * cfg.batch_size
        end_idx = start_idx + cfg.batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_preds[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

    logger.info(f"mask_count_min: {mask_count.min()}")
    mask_preds /= mask_count
    return valid_losses.avg, mask_preds


def get_optimizer(cfg: CFG, model: nn.Module) -> nn.optim.Optimizer:
    params = [
        {"params": model.encoder.parameters(), "lr": cfg.encoder_lr},
        {"params": model.decoder.parameters(), "lr": cfg.decoder_lr},
    ]
    if cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(params=params, weight_decay=cfg.weight_decay)
        return optimizer
    if cfg.optimizer == "RAdam":
        optimizer = optim.RAdam(params=params, weight_decay=1e-6)
        return optimizer
    raise ValueError(f"{cfg.optimizer} is not supported")


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


def get_scheduler(
    cfg: CFG, optimizer: nn.optim.Optimizer, step_per_epoch: int | None = None
) -> nn.optim.lr_scheduler._LRScheduler:
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epoch, eta_min=1e-6
        )
        return scheduler
    if cfg.scheduler == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            epochs=cfg.epoch,
            steps_per_epoch=step_per_epoch,
            max_lr=cfg.max_lr,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=1e2,
            final_div_factor=1e2,
        )
        return scheduler
    if cfg.scheduler == "TwoCyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=cfg.encoder_lr,
            max_lr=cfg.max_lr,
        )
    if cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg.T_0,
            T_mult=cfg.T_mult,
            eta_min=cfg.min_lr,
            last_epoch=-1,
        )
        return scheduler

    if cfg.scheduler == "CosineLRScheduler":
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=cfg.t_initial,
            lr_min=cfg.min_lr,
            warmup_lr_init=cfg.warmup_lr_init,
            warmup_t=cfg.warmup_t,
        )
        return scheduler

    if cfg.scheduler == "GradualWarmupScheduler":
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.T_max, eta_min=1e-7
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer=optimizer,
            multiplier=10,
            total_epoch=1,
            after_scheduler=scheduler_cosine,
        )
        return scheduler

    raise ValueError(f"Invalid scheduler: {cfg.scheduler}")


def get_loss(cfg: CFG) -> nn.Module:
    if cfg.loss == "BCEWithLogitsLoss":
        loss = nn.BCEWithLogitsLoss()
        return loss
    if cfg.loss == "DiceLoss":
        loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        return loss
    if cfg.loss == "LovaszLoss":
        loss = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        return loss
    if cfg.loss == "BCEDiceLoss":

        def _loss(y_pred, y_true, alpha=cfg.weight_bce):
            bce_loss = nn.BCEWithLogitsLoss()
            dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
            return alpha * bce_loss(y_pred, y_true) + (1 - alpha) * dice_loss(
                y_pred, y_true
            )

        return _loss
    if cfg.loss == "BCELovaszLoss":

        def _loss(y_pred, y_true, alpha=0.5):
            bce_loss = nn.BCEWithLogitsLoss()
            lobasz_loss = smp.losses.LovaszLoss(mode="binary", from_logits=True)
            return alpha * bce_loss(y_pred, y_true) + (1 - alpha) * lobasz_loss(
                y_pred, y_true
            )

        return _loss

    if cfg.loss == "BCEFocalLovaszLoss":

        def _loss(y_pred, y_true, alpha=cfg.weight_bce, beta=cfg.weight_focal):
            bce_loss = nn.BCEWithLogitsLoss()
            focal_loss = smp.losses.FocalLoss(mode="binary")
            lobasz_loss = smp.losses.LovaszLoss(mode="binary")
            return (
                (alpha * bce_loss(y_pred, y_true))
                + (beta * focal_loss(y_pred, y_true))
                + ((1 - alpha - beta) * lobasz_loss(y_pred, y_true))
            )

        return _loss

    if cfg.loss == "BCEFocalDiceLoss":

        def _loss(y_pred, y_true, alpha=cfg.weight_bce, beta=cfg.weight_focal):
            bce_loss = nn.BCEWithLogitsLoss()
            focal_loss = smp.losses.FocalLoss(mode="binary")
            dice_loss = smp.losses.DiceLoss(mode="binary")
            return (
                (alpha * bce_loss(y_pred, y_true))
                + (beta * focal_loss(y_pred, y_true))
                + ((1 - alpha - beta) * dice_loss(y_pred, y_true))
            )

        return _loss

    if cfg.loss == "BCEDiceLobaszLoss":

        def _loss(y_pred, y_true, alpha=0.7, beta=0.1):
            if alpha + beta > 1:
                raise ValueError("alpha + beta must be less than 1")
            bce_loss = nn.BCEWithLogitsLoss()
            dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
            lobasz_loss = smp.losses.LovaszLoss(mode="binary", from_logits=True)
            return (
                alpha * bce_loss(y_pred, y_true)
                + beta * lobasz_loss(y_pred, y_true)
                + (1 - alpha - beta) * dice_loss(y_pred, y_true)
            )

        return _loss

    if cfg.loss == "TverskyLoss":
        # alpha: FP weight, beta: FN weight
        loss = smp.losses.TverskyLoss(
            mode="binary", from_logits=True, alpha=0.7, beta=0.3
        )
        return loss

    if cfg.loss == "BCETverskyLoss":

        def _loss(y_pred, y_true, lamb=0.5):
            bce_loss = nn.BCEWithLogitsLoss()
            tversky_loss = smp.losses.TverskyLoss(
                mode="binary", from_logits=True, alpha=0.7, beta=0.3
            )
            return lamb * bce_loss(y_pred, y_true) + (1 - lamb) * tversky_loss(
                y_pred, y_true
            )

        return _loss

    if cfg.loss == "BCEDiceTverskyLoss":

        def _loss(y_pred, y_true):
            bce_loss = nn.BCEWithLogitsLoss()
            dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
            tversky_loss = smp.losses.TverskyLoss(
                mode="binary", from_logits=True, alpha=0.7, beta=0.3
            )
            return (
                bce_loss(y_pred, y_true)
                + dice_loss(y_pred, y_true)
                + 2 * tversky_loss(y_pred, y_true)
            )

        return _loss

    if cfg.loss == "SoftDice":
        loss = soft_dice_loss.SoftDiceLossV2()
        return loss

    raise ValueError(f"Invalid loss: {cfg.loss}")


def get_train_valid_loader(
    cfg: CFG,
    train_images: list[np.ndarray],
    train_labels: list[np.ndarray],
    valid_images: list[np.ndarray],
    valid_labels: list[np.ndarray],
) -> tuple[DataLoader, DataLoader]:
    train_dataset = VCDataset(
        cfg=cfg,
        images=train_images,
        labels=train_labels,
        phase="train",
        transform_fn=get_alb_transforms(cfg=cfg, phase="train"),
    )
    valid_dataset = VCDataset(
        cfg=cfg,
        images=valid_images,
        labels=valid_labels,
        phase="valid",
        transform_fn=get_alb_transforms(cfg=cfg, phase="valid"),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    # train_loader = PrefetchLoader(train_loader)
    # valid_loader = PrefetchLoader(train_loader)
    return train_loader, valid_loader


# training_fn
def train(cfg: CFG) -> None:
    for fold in range(1, cfg.n_fold + 1):
        seed_everything(seed=cfg.random_state)
        print("\n" + "=" * 30 + f" Fold {fold} " + "=" * 30 + "\n")
        (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            valid_xyxys,
        ) = get_train_valid_split(cfg=cfg, valid_id=fold)
        valid_xyxys = np.array(valid_xyxys)

        logger.info(f"train_images.shape: {len(train_images)}")
        logger.info(f"train_labels.shape: {len(train_labels)}")
        logger.info(f"valid_images.shape: {len(valid_images)}")
        logger.info(f"valid_labels.shape: {len(valid_labels)}")

        fragment_id = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}[fold]
        valid_mask_gt = cv2.imread(
            str(DATA_DIR / f"train/{fragment_id}/inklabels.png"), 0
        )
        valid_mask_gt = valid_mask_gt / 255
        pad0 = cfg.tile_size - valid_mask_gt.shape[0] % cfg.tile_size
        pad1 = cfg.tile_size - valid_mask_gt.shape[1] % cfg.tile_size
        valid_mask_gt = np.pad(valid_mask_gt, ((0, pad0), (0, pad1)), constant_values=0)

        net = VCNet(
            num_classes=1,
            arch=cfg.arch,
            encoder_name=cfg.encoder_name,
            in_chans=cfg.in_chans,
            weights=cfg.weights,
            aux_params=cfg.aux_params,
        )
        net = net.to(device=cfg.device, non_blocking=True)

        train_loader, valid_loader = get_train_valid_loader(
            cfg=cfg,
            train_images=train_images,
            train_labels=train_labels,
            valid_images=valid_images,
            valid_labels=valid_labels,
        )

        early_stopping = EarlyStopping(
            patience=cfg.patience,
            verbose=True,
            fold=str(fold),
            save_dir=OUTPUT_DIR / cfg.exp_name,
        )

        criterion = get_loss(cfg=cfg)
        criterion_cls = nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(cfg=cfg, model=net.model)
        scheduler = get_scheduler(
            cfg=cfg, optimizer=optimizer, step_per_epoch=len(train_loader)
        )
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=cfg.use_amp)

        best_score = 0

        use_awp = False

        for epoch in range(cfg.epoch):
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
            )
            valid_avg_loss, mask_preds = valid_per_epoch(
                cfg=cfg,
                model=net,
                valid_loader=valid_loader,
                criterion=criterion,
                fold=fold,
                epoch=epoch,
                valid_xyxys=valid_xyxys,
                valid_masks=valid_mask_gt,
            )
            scheduler.step()

            wandb.log(
                {
                    f"fold{fold}_train_avg_loss": train_avg_loss,
                    f"fold{fold}_valid_avg_loss": valid_avg_loss,
                }
            )

            best_dice, best_th = calc_cv(mask_gt=valid_mask_gt, mask_pred=mask_preds)
            score = best_dice
            wandb.log(
                {
                    f"fold{fold}_best_valid_dice": best_dice,
                    f"fold{fold}_best_valid_th": best_th,
                }
            )

            logger.info(
                f"Epoch {epoch} - train_avg_loss: {train_avg_loss} valid_avg_loss: {valid_avg_loss}"
                + f"best dice: {best_dice} best th: {best_th}"
            )
            if epoch > cfg.start_awp:
                if not use_awp:
                    logger.info(f"Start using awp at epoch {epoch}")
                use_awp = True

            if score > best_score:
                best_score = score
                torch.save(
                    {"preds": mask_preds, "best_dice": best_dice, "best_th": best_th},
                    str(CP_DIR / cfg.exp_name / f"best_fold{fold}.pth"),
                )

            early_stopping(val_loss=score, model=net)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        early_stopping.save_checkpoint(val_loss=0, model=net, prefix="last-")

        mask_preds = torch.load(CP_DIR / cfg.exp_name / f"best_fold{fold}.pth")["preds"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes[0].imshow(valid_mask_gt)
        axes[0].set_title("GT")
        axes[1].imshow(mask_preds)
        axes[1].set_title("Pred")
        axes[2].imshow((mask_preds >= best_th).astype(np.uint8))
        axes[2].set_title("Pred with threshold")
        fig.savefig(OUTPUT_DIR / cfg.exp_name / f"pred_mask_fold{fold}.png")

    best_dices = [
        torch.load(CP_DIR / cfg.exp_name / f"best_fold{fold}.pth")["best_dice"]
        for fold in range(1, cfg.n_fold + 1)
    ]
    logger.info(f"best dices: {best_dices}")
    mean_dice = np.mean(best_dices)
    logger.info("OOF mean dice: {}".format(mean_dice))
    wandb.log({"OOF mean dice": mean_dice, "best dices": best_dices})

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training has finished.")


# ====================================================
# Validation function
# ====================================================
def valid(cfg: CFG) -> None:
    all_preds = []
    all_masks = []
    # fragment id が 1 から始まるので 1 から始める
    for fold in range(1, cfg.n_fold + 1):
        seed_everything(seed=cfg.random_state)
        print("=" * 15 + f"{fold}" + "=" * 15)

        (
            _,
            _,
            valid_images,
            valid_labels,
            valid_xyxys,
        ) = get_train_valid_split(cfg=cfg, valid_id=fold)
        valid_xyxys = np.stack(valid_xyxys, axis=0)

        net = VCNet(
            num_classes=1,
            arch=cfg.arch,
            encoder_name=cfg.encoder_name,
            in_chans=cfg.in_chans,
        )
        net.load_state_dict(
            torch.load(OUTPUT_DIR / cfg.exp_name / f"checkpoint_{fold}.pth")
        )
        net = net.to(cfg.device)

        valid_dataset = VCDataset(
            cfg=cfg,
            images=valid_images,
            labels=valid_labels,
            phase="valid",
            transform_fn=get_alb_transforms(cfg=cfg, phase="valid"),
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=int(cfg.batch_size // 4),
            pin_memory=True,
            num_workers=cfg.num_workers - 2,
        )

        # (h, w)
        valid_mask_gt = cv2.imread(str(DATA_DIR / f"train/{fold}/inklabels.png"), 0)
        valid_mask_gt = valid_mask_gt / 255
        pad0 = cfg.tile_size - valid_mask_gt.shape[0] % cfg.tile_size
        pad1 = cfg.tile_size - valid_mask_gt.shape[1] % cfg.tile_size
        valid_mask_gt = np.pad(valid_mask_gt, ((0, pad0), (0, pad1)), constant_values=0)

        valid_labels = np.stack(valid_labels, axis=0)
        mask_preds = np.zeros(valid_mask_gt.shape)
        dbg(f"valid_mask_gt.shape: {valid_mask_gt.shape}")
        mask_count = np.zeros(valid_mask_gt.shape)
        valid_preds = []
        valid_targets = []
        for step, (image, target) in tqdm(
            enumerate(valid_loader), total=len(valid_loader), dynamic_ncols=True
        ):
            net.eval()
            image = image.to(cfg.device)
            target = target.to(cfg.device)

            with torch.inference_mode():
                y_preds = net(image)

                valid_preds.append(y_preds.to("cpu").detach().numpy())
                valid_targets.append(target.to("cpu").detach().numpy())

            y_preds = torch.sigmoid(y_preds).to("cpu").detach().numpy()

            start_idx = step * cfg.batch_size
            end_idx = start_idx + cfg.batch_size
            for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
                mask_preds[y1:y2, x1:x2] = y_preds[i].squeeze(0)
                mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

        auc = roc_auc_score(valid_labels.reshape(-1), mask_preds.reshape(-1))
        logger.info("RoC-Auc: ", auc)
        wandb.log({f"valid_fold{fold}_RoC_AUC_full": auc})
        all_masks.append(valid_labels.reshape(-1))
        all_preds.append(mask_preds.reshape(-1))

    flat_preds = np.hstack(all_preds).reshape(-1).astype(np.float32)
    # flat_masks = (np.hstack(all_masks).reshape(-1) / 255).astype(np.int8)

    plt.hist(flat_preds, bins=50)
    plt.savefig(OUTPUT_DIR / cfg.exp_name / "flat_pred_hist.png")


# ====================================================
# test functions
# ====================================================
def read_image(cfg: CFG, fragment_id: str) -> np.ndarray:
    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2 + 1
    idxs = range(start, end)

    for i in tqdm(idxs):
        image = cv2.imread(
            str(DATA_DIR / f"test/{fragment_id}/surface_volume/{i:02}.tif"), 0
        )

        pad0 = cfg.tile_size - image.shape[0] % cfg.tile_size
        pad1 = cfg.tile_size - image.shape[1] % cfg.tile_size

        # 画像の下と右にpadを追加
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    images = np.stack(images, axis=2)
    return images


def make_test_datast(cfg: CFG, fragment_id: str) -> tuple[DataLoader, np.ndarray]:
    """test用のdatasetとloaderを作成する
    Args:
        cfg (CFG): config
        fragment_id (int): fragment_id
    Returns:
        tuple[DataLoader, np.ndarray]: test用のloaderとタイル座標
    """
    test_images = read_image(cfg=cfg, fragment_id=fragment_id)
    dbg(f"test_images.shape: {test_images.shape}")

    # tile_sizeで何枚に分割できるか, x方向とy方向でそれぞれ計算
    # x1, y1は左上の座標なのでtile_sizeを引く
    y1_list = list(range(0, test_images.shape[0] - cfg.tile_size + 1, cfg.stride))
    x1_list = list(range(0, test_images.shape[1] - cfg.tile_size + 1, cfg.stride))

    dbg(f"len(y1_list): {len(y1_list)}, {y1_list[:5]}")
    dbg(f"len(x1_list): {len(x1_list)}, {x1_list[:5]}")

    test_image_list = []
    xyxy_list = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + cfg.tile_size
            x2 = x1 + cfg.tile_size
            test_image_list.append(test_images[y1:y2, x1:x2])
            xyxy_list.append((x1, y1, x2, y2))

    xyxy_list = np.stack(xyxy_list)
    test_dataset = VCDataset(
        cfg=cfg,
        images=test_image_list,
        labels=None,
        phase="test",
        transform_fn=get_alb_transforms(cfg=cfg, phase="test"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader, xyxy_list


# predict function
def predict(cfg: CFG, test_data_dir: Path, threshold: float) -> np.ndarray:
    fragment_ids = list(DATA_DIR.rglob("test/*"))
    model = build_ensemble_model(cfg=cfg)
    if cfg.use_tta:
        model = tta.SegmentationTTAWrapper(model, cfg.tta_transforms, merge_mode="mean")
    for fragment_id, fragment_path in enumerate(fragment_ids, start=1):
        test_loader, xyxy_list = make_test_datast(
            cfg=cfg, fragment_id=fragment_path.stem
        )
        dbg(f"xyxy_list.shape: {xyxy_list.shape}")
        dbg(f"xyxy_list.min: {xyxy_list.min()}, xyxy_list.max: {xyxy_list.max()}")

        binary_mask = cv2.imread(
            str(DATA_DIR / f"test/{fragment_path.stem}/mask.png"), 0
        )
        binary_mask = (binary_mask / 255).astype(np.int8)

        ori_h = binary_mask.shape[0]
        ori_w = binary_mask.shape[1]

        logger.info(f"ori_h: {ori_h}, ori_w: {ori_w}")

        pad0 = cfg.tile_size - binary_mask.shape[0] % cfg.tile_size
        pad1 = cfg.tile_size - binary_mask.shape[1] % cfg.tile_size

        binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask_pred = np.zeros(binary_mask.shape)
        mask_pred_count = np.zeros(binary_mask.shape)

        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(cfg.device)
            batch_size = images.size(0)

            with torch.inference_mode():
                y_preds = model(images).numpy()

            start_idx = step * cfg.batch_size
            end_idx = start_idx + batch_size
            for i, (x1, y1, x2, y2) in enumerate(xyxy_list[start_idx:end_idx]):
                mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
                mask_pred_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))

    logger.info(f"mask_pred_count: {mask_pred_count.min()}, {mask_pred_count.max()}")
    mask_pred /= mask_pred_count

    mask_pred = mask_pred[:ori_h, :ori_w]
    binary_mask = binary_mask[:ori_h, :ori_w]
    mask_pred = (mask_pred > threshold).astype(np.int8)
    mask_pred = mask_pred * binary_mask

    if IS_TRAIN:
        logger.info("save mask_pred_count, mask_pred")
        plt.imsave(
            str(OUTPUT_DIR / cfg.exp_name / "mas-pred-count.png"), mask_pred_count
        )
        # plt.imsave(OUTPUT_DIR / cfg.exp_name / "mas-pred.png", mask_pred)
        plt.close("all")
    else:
        plt.imshow(mask_pred_count)
        plt.imshow(mask_pred)
        plt.imshow(binary_mask)
        plt.show()
        plt.close("all")
    return mask_pred


def test(cfg: CFG, threshold: float = 0.4) -> pd.DataFrame:
    test_data_root = DATA_DIR / "test"
    test_data_path = list(test_data_root.glob("*"))
    preds = []
    for fp in tqdm(test_data_path, total=len(test_data_path)):
        print(fp)
        pred_tile_image = predict(cfg=cfg, test_data_dir=fp, threshold=threshold)

        if IS_TRAIN:
            plt.imsave(
                str(
                    OUTPUT_DIR
                    / cfg.exp_name
                    / f"test_pred_tile_image_{str(fp).replace('/', '_')}.png"
                ),
                pred_tile_image,
            )
            plt.close("all")
            plt.imsave(
                str(
                    OUTPUT_DIR
                    / cfg.exp_name
                    / f"test_pred_tile_image_mask_{str(fp).replace('/', '_')}.png"
                ),
                np.where(pred_tile_image > threshold, 1, 0),
            )
            plt.close("all")
        else:
            plt.imshow(pred_tile_image)
            plt.title("pred_tile_image")
            plt.show()
            plt.imshow(np.where(pred_tile_image > threshold, 1, 0))
            plt.title("pred_tile_mask")
            plt.show()

        logger.info("start to process rle")
        # starts_idx, lengths = rle(pred_tile_image.copy(), threshold=threshold)
        # dbg(f"starts_idx: {starts_idx[:10]}, lengths: {lengths[:10]}")
        # inklabels_rle = " ".join(map(str, sum(tqdm(zip(starts_idx, lengths)), ())))
        inklabels_rle = fast_rle(img=pred_tile_image)
        if IS_TRAIN:
            logger.info(
                f"ID: {str(fp).split('/')[-1]}, inklabels_rle: {inklabels_rle[:10]}"
            )
        preds.append({"Id": str(fp).split("/")[-1], "Predicted": inklabels_rle})
        del pred_tile_image, inklabels_rle
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(preds)


# =======================================================================
# main part
# =======================================================================
def main() -> None:
    cfg = CFG()
    logger.info(f"{cfg.__dict__}")
    seed_everything(seed=cfg.random_state)
    (OUTPUT_DIR / cfg.exp_name).mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    logger.info(f"Start Time = {start_time.strftime('%Y-%m-%d-%H-%M-%S')}")

    if IS_TRAIN and not SKIP_TRAIN:
        # if False:
        wandb.init(
            project="vesuvius_challenge",
            config=asdict(cfg),
            group=f"{cfg.arch}_{cfg.encoder_name}",
            name=f"{cfg.exp_name}_{start_time}",
        )
        train(cfg=cfg)
        # valid(cfg=cfg)
        wandb.finish()
        train_duration = datetime.now() - start_time
        logger.info(f"Train Duration = {train_duration}")
        with (OUTPUT_DIR / cfg.exp_name / "config.json").open("w") as f:
            json.dump(cfg.__dict__, f, indent=4)

    if MAKE_SUB:
        preds = test(cfg, threshold=THR)
        print(preds)
        save_path = (
            "submission.csv"
            if not IS_TRAIN
            else OUTPUT_DIR / cfg.exp_name / "submission.csv"
        )
        preds.to_csv(save_path, index=False)
        if SKIP_TRAIN:
            test_duration = datetime.now() - start_time
        else:
            test_duration = datetime.now() - start_time - train_duration
        logger.info(f"Test Duration = {test_duration}")

    total_duration = datetime.now() - start_time
    logger.info(f"Total Duration = {total_duration}")


if __name__ == "__main__":
    main()
