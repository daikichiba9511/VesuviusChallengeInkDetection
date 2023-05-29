"""exp055
新しいbaselineを作る

Ref:
[1] https://www.kaggle.com/code/hengck23/lb0-68-one-fold-stacked-unet
"""
from __future__ import annotations

import gc
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.functional as F
import torch.nn as nn
import transformers
from einops import rearrange, reduce, repeat
from loguru import logger
from PIL import Image
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock, UnetDecoder
from timm.models.resnet import resnet10t, resnet34d
from torch.utils.data import DataLoader, Dataset

import wandb

IS_TRAIN = not Path("/kaggle/working").exists()
MAKE_SUB: bool = False
SKIP_TRAIN = False

if IS_TRAIN:
    from src.augmentations import cutmix
    from src.train_utils import (
        EarlyStopping,
        get_alb_transforms,
        get_loss,
        get_optimizer,
        get_scheduler,
        seed_everything,
        train_per_epoch,
        valid_per_epoch,
    )

dbg = logger.debug

# ssl._create_default_https_context = ssl._create_unverified_context
warnings.simplefilter("ignore")

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


@dataclass
class CFG:
    expname: str = "exp055"
    mode = ["train", "test"]

    # --- Data Config
    fragment_z: tuple[int, int] = (29, 37)
    """should be 0 ~ 65, [z_min, z_max)"""
    crop_size: int = 384
    """same as tile size."""
    crop_fade: int = 56
    """画像の端の部分を切り取るときに、端の部分をfadeさせることで、画像の端の部分の情報を失わないようにする。
    """
    crop_depth: int = 5

    # --- Infer Config
    is_tta: bool = True
    ink_threshold: float = 0.5
    checkpoints: list[str] = field(default_factory=lambda: [""])


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
    if cfg.in_chans % 2 == 0:
        end = mid + cfg.in_chans // 2
    else:
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


def read_data(mode: list[str], fragment_id: int, z0: int, z1: int) -> Data:
    volume = []
    for i in range(z0, z1):
        v_i = np.array(
            Image.open(DATA_DIR / f"{fragment_id}" / "surface_volume" / f"{i:02d}.tif"),
            dtype=np.uint16,
        )
        v_i = (v_i >> 8).astype(np.uint8)
        volume.append(v_i)
        logger.info(f"read_data: volume-{i:02d} {v_i.shape} {v_i.dtype}")

    volume = np.stack(volume, axis=-1)
    height, width, depth = volume.shape

    mask = cv2.imread(f"{DATA_DIR}/{fragment_id}/mask.png", cv2.IMREAD_GRAYSCALE)
    mask = binarize(mask)

    if "train" in mode:
        ir = cv2.imread(f"{DATA_DIR}/{fragment_id}/ir.png", cv2.IMREAD_GRAYSCALE)
        ir = ir / 255
        label = cv2.imread(
            f"{DATA_DIR}/{fragment_id}/inklabels.png", cv2.IMREAD_GRAYSCALE
        )
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
    if fragment_id == "2a":
        y = 9456
        data = read_data(cfg.mode, 2, cfg.fragment_z[0], cfg.fragment_z[1])
        assert "train" in cfg.mode
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
        data = read_data(cfg.mode, 2, cfg.fragment_z[0], cfg.fragment_z[1])
        assert "train" in cfg.mode
        data = Data(
            volume=data.volume[y:, :, :],
            fragment_id=2,
            label=data.label[y:, :],
            mask=data.mask[y:, :],
            ir=data.ir[y:, :],
        )
        return data

    data = read_data(cfg.mode, fragment_id, cfg.fragment_z[0], cfg.fragment_z[1])
    return data


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
        batch_size=cfg.valid_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    # train_loader = PrefetchLoader(train_loader)
    # valid_loader = PrefetchLoader(train_loader)
    return train_loader, valid_loader


# =================================================
# Model
# =================================================
class SmpUnetDecorder(nn.Module):
    def __init__(
        self, in_channels: list[int], out_channels: list[int], skip_channels: list[int]
    ) -> None:
        super().__init__()
        self.center = nn.Identity()

        in_channels = [
            in_channels,
        ] + out_channels[:-1]
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=in_channel,
                    skip_channels=skip_channel,
                    out_channels=out_channel,
                    use_batchnorm=True,
                    attention_type=None,
                )
                for in_channel, skip_channel, out_channel in zip(
                    in_channels, skip_channels, out_channels
                )
            ]
        )

    def forward(
        self, feature: torch.Tensor, skip: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature (torch.Tensor): (batch_size, in_channels, height, width)
            skip (list[torch.Tensor]): [(batch_size, skip_channels, height, width), ...]

        Returns:
            last (torch.Tensor): (batch_size, out_channels[-1], height, width)
            decode (list[torch.Tensor]): [(batch_size, out_channels[i], height, width), ...]
        """
        dense = self.center(feature)
        decode = []
        for i, block in enumerate(self.blocks):
            dense = block(dense, skip[i])
            decode.append(dense)
        last = dense
        return last, decode


class VCNet(nn.Module):
    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_type = ["inference", "loss"]

        conv_dim = 64
        encoder1_dims = [conv_dim, 64, 128, 256, 512]
        decoder1_dims = [256, 128, 64, 64]

        self.encoder1 = resnet34d(pretrained=False, in_channels=cfg.crop_depth)
        self.decoder1 = SmpUnetDecorder(
            in_channels=encoder1_dims[-1],
            skip_channels=encoder1_dims[:-1][::-1],
            out_channels=decoder1_dims,
        )

        # pool attention weights
        self.weights1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, stride=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for dim in encoder1_dims
            ]
        )
        self.logit1 = nn.Conv2d(
            in_channels=decoder1_dims[-1], out_channels=1, kernel_size=1
        )

        encoder2_dims = [64, 128, 256, 512]
        decoder2_dims = [128, 64, 32]
        self.encoder2 = resnet10t(pretrained=False, in_channels=decoder1_dims[-1])
        self.decoder2 = SmpUnetDecorder(
            in_channels=encoder2_dims[-1],
            skip_channels=encoder2_dims[:-1][::-1],
            out_channels=decoder2_dims,
        )
        self.logit2 = nn.Conv2d(decoder2_dims[-1], 1, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]):
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
        vv = [volume[:, i : i + self.cfg.crop_depth] for i in [0, 2, 4]]
        K = len(vv)
        x = torch.cat(vv, dim=0)

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

        for i in range(len(encoder)):
            e = encoder[i]
            f = self.weights1[i](e)
            _, c, h, w = e.shape
            f = rearrange(f, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)
            e = rearrange(e, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)
            w = F.softmax(f, dim=1)
            e = (w * e).sum(dim=1)
            encoder[i] = e

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last, decoder = self.decoder1(feature, skip)

        x = last
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
        last, decoder = self.decoder2(feature, skip)
        logits2 = self.logit2(last)
        logits2 = F.interpolate(
            logits2, size=(H, W), mode="bilinear", align_corners=False
        )

        output = {
            "ink": torch.sigmoid(logits2),
        }
        return output


# =================================================
# train funcions
# =================================================
def training_fn(cfg: CFG) -> None:
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

    score = (
        beta * beta / (1 + beta * beta) * 1 / recall
        + 1 / (1 + beta * beta) * 1 / precision
    )
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
        p_sum, precision, recall, fpr, dice, score = calc_metric(
            thr, ink, label, mask_sum
        )
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
        p_sum, precision, recall, fpr, dice, score = calc_metric(
            thr, ink, label, mask_sum
        )
        text.append(
            f"{p_sum:.4f} {thr:.2f} {precision:.4f} {recall:.4f} {fpr:.4f} {dice:.4f} {score:.4f}"
        )

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


def tta_rotate(net: VCNet, volume: torch.Tensor) -> torch.Tensor:
    B, _, H, W = volume.shape
    rotated_volumes = [
        volume,
        torch.rot90(volume, k=1, dims=(-2, -1)),
        torch.rot90(volume, k=2, dims=(-2, -1)),
        torch.rot90(volume, k=3, dims=(-2, -1)),
    ]
    K = len(rotated_volumes)
    batch = {
        "volume": torch.cat(rotated_volumes, dim=0),
    }
    output = net(batch)
    ink = output["ink"]
    ink = ink.reshape(K, B, 1, H, W)
    ink = [
        ink[0],
        torch.rot90(ink[1], k=-1, dims=(-2, -1)),
        torch.rot90(ink[1], k=-2, dims=(-2, -1)),
        torch.rot90(ink[1], k=-3, dims=(-2, -1)),
    ]
    ink = torch.stask(ink, dim=0).mean(dim=0)
    return ink


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


def infer(
    cfg: CFG, fragement_ids: list[str], checkpoints: list[str | Path]
) -> pd.DataFrame:
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
            text = metric_to_text(ink_pred, data.label, data.mask)
            logger.info(text)

    sub_df = pd.DataFrame.from_dict(sub)
    return sub_df


def main() -> None:
    cfg = CFG()
    if "train" in cfg.mode:
        pass

    elif "test" in cfg.mode:
        test_fragments_ids = get_test_fragments_ids(cfg)
        if is_skip_test():
            sub_df = pd.DataFrame(
                {"Id": test_fragments_ids, "Predicted": ["1 2", "1 2"]}
            )
        else:
            sub_df = infer(cfg, test_fragments_ids, cfg.checkpoints)
            print(sub_df)
        sub_df.to_csv("submission.csv", index=False)
        logger.info("make sub finish")


if __name__ == "__main__":
    main()
