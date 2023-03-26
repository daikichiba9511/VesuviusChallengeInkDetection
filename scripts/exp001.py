"""exp001

baseline

Reference:
[1]
https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code
"""
from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

dbg = logger.debug

# torchvision.disable_beta_transforms_warning()
warnings.simplefilter("ignore")


IS_TRAIN = not Path("/kaggle/working").exists()
print(IS_TRAIN)

MAKE_SUB: bool = True


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


# ==============================================================
# Config
# ==============================l================================
@dataclass(frozen=True)
class CFG:
    exp_name = "exp000"

    n_fold = 5
    random_state = 42
    lr = 1e-5
    max_lr = 1e-5
    patience = 15
    n_fold = 3
    epoch = 8
    batch_size = 8
    image_size = (512, 512)
    num_workers = mp.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===============================================================
# utils
# ===============================================================
def rle(image: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        image: 画像, (H, W, 3)
        threshold: 閾値
    """
    flat_image = image.flatten()
    flat_image = np.where(flat_image > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_image[:-1] == 0) & (flat_image[1:] == 1))
    ends = np.array((flat_image[:-1] == 1) & (flat_image[1:] == 0))
    starts_idx = np.where(starts)[0] + 2
    ends_idx = np.where(ends)[0] + 2
    length = ends_idx - starts_idx
    return starts_idx, length


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
    print(f"{image1.shape =}, {image2.shape =}, {image3.shape =}")
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


def get_train_transform_fn(image_size: tuple[int, int]) -> transforms.Compose:
    """
    Args:
        image_size: (H, W)
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(
                size=image_size, antialias=True, max_size=None
            ),  # If max_size=None, ignore aspect ratio when resizing
        ]
    )


def get_test_transform_fn(image_size: tuple[int, int]) -> transforms.Compose:
    """
    Args:
        image_size: (H, W)
    """
    return transforms.Compose(
        [
            transforms.Resize(size=image_size, antialias=True, max_size=None),
        ]
    )


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
        labels: list[np.ndarray] | None = None,
        phase: str = "train",
        crop_size: int = 256,
        transform_fn: Callable | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.phase = phase
        self.crop_size = crop_size
        self.transform_fn = transform_fn

        self.cell_counts = [
            (
                math.ceil(image.shape[1] / self.crop_size)
                * math.ceil(image.shape[2] / self.crop_size)
            )
            for image in images
        ]

        self.data_count = 0
        if phase == "train":
            if self.labels is None:
                raise ValueError("Expected self.labels is not None, but got None")

            self.cell_id_maps = {}
            counter = 0
            for image_num, image in enumerate(self.images):
                # crop_sizeで画像を何分割できるか
                cell_count = math.ceil(image.shape[1] / self.crop_size) * math.ceil(
                    image.shape[2] / self.crop_size
                )
                for cell_id in range(cell_count):
                    h_cell_idx = cell_id // math.ceil(
                        self.labels[image_num].shape[1] / self.crop_size
                    )
                    w_cell_idx = cell_id - (
                        (
                            h_cell_idx
                            * math.ceil(
                                self.labels[image_num].shape[1] / self.crop_size
                            )
                        )
                    )
                    cropped_image = self.labels[image_num][
                        h_cell_idx * self.crop_size : (h_cell_idx + 1) * self.crop_size,
                        w_cell_idx * self.crop_size : (w_cell_idx + 1) * self.crop_size,
                    ]
                    if cropped_image.sum() == 0:
                        continue
                    self.data_count += 1
                    self.cell_id_maps[counter] = (image_num, cell_id)
                    counter += 1
        else:
            for image in self.images:
                self.data_count += math.ceil(
                    image.shape[1] / self.crop_size
                ) * math.ceil(image.shape[2] / self.crop_size)

    def __len__(self) -> int:
        return self.data_count

    def calc_image_num(self, idx: int) -> tuple[int, int]:
        """
        Returns:
            (i, idx - (cum_cell_count - cell_count))
        """
        cum_cell_count = 0
        for i, cell_count in enumerate(self.cell_counts):
            cum_cell_count += cell_count
            if idx + 1 <= cum_cell_count:
                return i, idx - (cum_cell_count - cell_count)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (image, mask)
        """
        if self.phase == "train":
            image_num, cell_id = self.cell_id_maps[idx]
        else:
            image_num, cell_id = self.calc_image_num(idx=idx)

        target_image = self.images[image_num]
        target_image = np.moveaxis(target_image, 0, 2)
        if self.phase != "test":
            if self.labels is None:
                ValueError("Expected not None, but got None")
            # (h, w)
            target_label = self.labels[image_num]

        h_cell_num = cell_id // math.ceil(target_image.shape[1] / self.crop_size)
        w_cell_num = cell_id - (
            h_cell_num * math.ceil(target_image.shape[1] / self.crop_size)
        )

        # (h, w, 65)
        cropped_image = target_image[
            h_cell_num * self.crop_size : (h_cell_num + 1) * self.crop_size,
            w_cell_num * self.crop_size : (w_cell_num + 1) * self.crop_size,
        ]

        if self.phase in ["train", "valid"]:
            cropped_label = target_label[
                h_cell_num * self.crop_size : (h_cell_num + 1) * self.crop_size,
                w_cell_num * self.crop_size : (w_cell_num + 1) * self.crop_size,
            ]
            if self.transform_fn is not None:
                masks = torch.tensor(cropped_label)
                cropped_image = self.transform_fn(
                    torch.tensor(cropped_image).permute(2, 0, 1)
                )
                masks = F.resize(
                    masks.unsqueeze(0),
                    size=[cropped_image.shape[1], cropped_image.shape[2]],
                ).squeeze(0)
                # dbg(f"{cropped_image.shape =}, {cropped_label.shape}")
            image = cropped_image
        else:
            if self.transform_fn is not None:
                cropped_image = self.transform_fn(
                    torch.tensor(cropped_image).permute(2, 0, 1),
                )
            image = cropped_image
            masks = -1

        # Cast for training / inference to adjust dtype of model weight
        # (65, H, W)
        image_tensor = torch.tensor(image).to(dtype=torch.float32)
        mask_tensor = torch.tensor(masks / 255.0).to(dtype=torch.float32)
        assert image_tensor.shape == (65, 512, 512), (
            f"{image_tensor.shape =},"
            + f" {image.shape =},"
            + f"{cropped_image.shape =}"
        )
        return image_tensor, mask_tensor


# =======================================================================
# Model
# =======================================================================
class VCNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.contracting_11 = self._build_conv_block(in_channels=65, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self._build_conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self._build_conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self._build_conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self._build_conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_12 = self._build_conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_22 = self._build_conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_32 = self._build_conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_42 = self._build_conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _build_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, 64, 256, 256)
        contracting_11_out = self.contracting_11(x)
        # (N, 64, 128, 128)
        contracting_12_out = self.contracting_12(contracting_11_out)
        # (N, 128, 128, 128)
        contracting_21_out = self.contracting_21(contracting_12_out)
        # (N, 128, 64, 64)
        contracting_22_out = self.contracting_22(contracting_21_out)
        # (N, 256, 64, 64)
        contracting_31_out = self.contracting_31(contracting_22_out)
        # (N, 256, 32, 32)
        contracting_32_out = self.contracting_32(contracting_31_out)
        # (N, 512, 32, 32)
        contracting_41_out = self.contracting_41(contracting_32_out)
        # (N, 512, 16, 16)
        contracting_42_out = self.contracting_42(contracting_41_out)
        # (N, 1024, 16, 16)
        middle_out = self.middle(contracting_42_out)
        # (N, 512, 32, 32)
        expansive_11_out = self.expansive_11(middle_out)
        # (N, 1024, 32, 32) -> (N, 512, 32, 32)
        expansive_12_out = self.expansive_12(
            torch.cat([expansive_11_out, contracting_41_out], dim=1)
        )
        # (N, 256, 64, 64)
        expansive_21_out = self.expansive_21(expansive_12_out)
        # (N, 512, 64, 64) -> (N, 256, 64, 64)
        expansive_22_out = self.expansive_22(
            torch.cat([expansive_21_out, contracting_31_out], dim=1)
        )
        # (N, 128, 128, 128)
        expansive_31_out = self.expansive_31(expansive_22_out)
        # (N, 256, 128, 128) -> (N, 128, 128, 128)
        expansive_32_out = self.expansive_32(
            torch.cat([expansive_31_out, contracting_21_out], dim=1)
        )
        # (N, 64, 256, 256)
        expansive_41_out = self.expansive_41(expansive_32_out)
        # (N, 128, 256, 256) -> (N, 64, 256, 256)
        expansive_42_out = self.expansive_42(
            torch.cat([expansive_41_out, contracting_11_out], dim=1)
        )
        output = self.output(expansive_42_out)
        return output


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
        score = -val_loss
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


# ==========================================================
# training function
# ==========================================================
def train(
    cfg: CFG,
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    data_fold = split_cv(images=images, labels=labels, masks=masks)
    for fold in range(cfg.n_fold):
        print("\n" + "=" * 30 + f" Fold {fold} " + "=" * 30 + "\n")

        net = VCNet(num_classes=1)
        net.to(device=cfg.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-2)

        train_dataset = VCDataset(
            images=data_fold[fold]["train_image"],
            labels=data_fold[fold]["train_label"],
            phase="train",
            crop_size=cfg.image_size[0],
            transform_fn=get_train_transform_fn(image_size=cfg.image_size),
        )
        valid_dataset = VCDataset(
            images=data_fold[fold]["valid_image"],
            labels=data_fold[fold]["valid_label"],
            phase="valid",
            crop_size=cfg.image_size[0],
            transform_fn=get_train_transform_fn(image_size=cfg.image_size),
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            pin_memory=True,
            # pin_memory=False,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.num_workers,
            # num_workers=1,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.batch_size,
            pin_memory=True,
            # pin_memory=False,
            shuffle=False,
            num_workers=cfg.num_workers,
            # num_workers=1,
        )
        early_stopping = EarlyStopping(
            patience=cfg.patience,
            verbose=True,
            fold=str(fold),
            save_dir=OUTPUT_DIR / cfg.exp_name,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            epochs=cfg.epoch,
            steps_per_epoch=len(train_loader),
            max_lr=cfg.max_lr,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=1e3,
            final_div_factor=1e3,
        )

        valid_metrics = []
        learning_rates = []
        for epoch in range(cfg.epoch):
            running_loss = 0.0
            # train_losses = []
            with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
                for i, (image, target) in pbar:
                    net.train()

                    image = image.to(cfg.device)
                    target = target.to(cfg.device)

                    outputs = net(image)

                    loss = criterion(outputs.squeeze(), target)

                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

                    running_loss += loss.item()
                    pbar.set_postfix(
                        {"epoch": f"{epoch}", "loss": f"{loss.item():.4f}"}
                    )
                    learning_rates.append(optimizer.param_groups[0]["lr"])
                    scheduler.step()

            valid_preds = []
            valid_targets = []
            for i, (image, target) in tqdm(
                enumerate(valid_loader),
                total=len(valid_loader),
                smoothing=0,
                dynamic_ncols=True,
            ):
                net.eval()
                image = image.to(cfg.device)
                target = target.to(cfg.device)

                with torch.inference_mode():
                    outputs = net(image)
                    outputs = outputs.sigmoid()
                    valid_preds.append(outputs.to("cpu").detach().numpy())
                    valid_targets.append(target.to("cpu").detach().numpy())

            # 端を切る
            w_count = math.ceil(
                data_fold[fold]["valid_label"][0].shape[1] / cfg.image_size[1]
            )
            h_count = math.ceil(
                data_fold[fold]["valid_label"][0].shape[0] / cfg.image_size[0]
            )

            tile_array = make_tile_array(
                valid_preds=valid_preds,
                h_count=h_count,
                w_count=w_count,
                image_size=cfg.image_size,
            )

            pred_tile_image = concat_tile(tile_array)
            pred_tile_image = np.where(
                data_fold[fold]["valid_mask"][0] > 1,
                pred_tile_image[
                    : data_fold[fold]["valid_label"][0].shape[0],
                    : data_fold[fold]["valid_label"][0].shape[1],
                ],
                0,
            )
            auc = roc_auc_score(
                data_fold[fold]["valid_label"][0].reshape(-1),
                pred_tile_image.reshape(-1),
            )
            print(f"AUC: {auc:.5f}")
            lr = optimizer.param_groups[0]["lr"]
            valid_metrics.append(lr)
            early_stopping(-auc, net)

        early_stopping.save_checkpoint(val_loss=-auc, model=net, prefix="last-")
        # if early_stopping.early_stop:
        #     print("Detected Early Stopping")
        #     break

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(learning_rates)
    ax2 = ax1.twinx()
    ax2.plot(valid_metrics)
    plt.savefig(str(OUTPUT_DIR / cfg.exp_name / "metrics_lr.png"))

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training has finished.")


# ====================================================
# Validation function
# ====================================================
def valid(cfg: CFG, data_fold: list[dict]) -> None:
    all_preds = []
    all_masks = []
    for fold in range(cfg.n_fold):
        print("=" * 15 + f"{fold}" + "=" * 15)

        net = VCNet(num_classes=1)
        net.load_state_dict(
            torch.load(OUTPUT_DIR / cfg.exp_name / f"checkpoint_{fold}.pth")
        )
        net.to(cfg.device)

        valid_dataset = VCDataset(
            images=data_fold[fold]["valid_image"],
            labels=data_fold[fold]["valid_label"],
            phase="valid",
            crop_size=cfg.image_size[0],
            transform_fn=get_train_transform_fn(image_size=cfg.image_size),
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=int(cfg.batch_size // 4),
            pin_memory=True,
            num_workers=cfg.num_workers - 2,
        )

        valid_preds = []
        valid_targets = []
        for i, (image, target) in tqdm(
            enumerate(valid_loader), total=len(valid_loader), dynamic_ncols=True
        ):
            net.eval()
            image = image.to(cfg.device)
            target = target.to(cfg.device)

            with torch.inference_mode():
                outputs = net(image)
                outputs = outputs.sigmoid()

                valid_preds.append(outputs.to("cpu").detach().numpy())
                valid_targets.append(target.to("cpu").detach().numpy())

        # 端を切る
        w_count = math.ceil(
            data_fold[fold]["valid_label"][0].shape[1] / cfg.image_size[1]
        )
        h_count = math.ceil(
            data_fold[fold]["valid_label"][0].shape[0] / cfg.image_size[0]
        )

        tile_array = make_tile_array(
            valid_preds=valid_preds,
            h_count=h_count,
            w_count=w_count,
            image_size=cfg.image_size,
        )
        pred_tile_image = concat_tile(tile_array)
        pred_tile_image = np.where(
            data_fold[fold]["valid_mask"][0] > 1,
            pred_tile_image[
                : data_fold[fold]["valid_label"][0].shape[0],
                : data_fold[fold]["valid_label"][0].shape[1],
            ],
            0,
        )
        auc = roc_auc_score(
            data_fold[fold]["valid_label"][0].reshape(-1), pred_tile_image.reshape(-1)
        )
        print("Auc: ", auc)
        all_masks.append(data_fold[fold]["valid_label"][0].reshape(-1))
        all_preds.append(pred_tile_image.reshape(-1))

    flat_preds = np.hstack(all_preds).reshape(-1).astype(np.float32)
    flat_masks = (np.hstack(all_masks).reshape(-1) / 255).astype(np.int8)

    plt.hist(flat_preds, bins=50)
    plt.savefig(OUTPUT_DIR / cfg.exp_name / "flat_pred_hist.png")

    thr_list = []
    for thr in np.arange(0.2, 0.6, 0.1):
        _val_pred = np.where(flat_preds > thr, 1, 0).astype(np.int8)
        score = f1_score(flat_masks, _val_pred)
        print(f"Threshold: {thr} --> Score: {score}")
        thr_list.append({"thr": thr, "score": score})


# ====================================================
# test functions
# ====================================================
def predict(cfg: CFG, test_data_dir: Path) -> np.ndarray:
    test_image = []
    for i in tqdm(range(65)):
        test_image.append(
            cv2.imread(str(test_data_dir / "surface_volume" / f"{i:02}.tif"), 0)
        )

    test_image = np.stack(test_image)
    print(f"{test_image.shape}")

    test_mask = cv2.imread(str(test_data_dir / "mask.png"), 0)

    nets = []
    for fold in range(cfg.n_fold):
        net = VCNet(num_classes=1)
        net.to(cfg.device)
        net.load_state_dict(
            torch.load(OUTPUT_DIR / cfg.exp_name / f"checkpoint_{fold}.pth")
        )
        nets.append(net)

    test_dataset = VCDataset(
        images=[test_image],
        phase="test",
        crop_size=cfg.image_size[0],
        transform_fn=get_test_transform_fn(image_size=cfg.image_size),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=int(cfg.batch_size // 8),
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    test_preds = []
    for i, (image, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        net.eval()
        image = image.to(cfg.device)
        target = target.to(cfg.device)

        with torch.inference_mode():
            outputs_all = np.zeros((image.shape[0], image.shape[2], image.shape[3]))
            for net in nets:
                outputs = net(image)
                outputs = outputs.sigmoid()
                outputs_all += outputs.squeeze().to("cpu").detach().numpy()
            test_preds.append(outputs_all)

    w_count = math.ceil(test_image[0].shape[1] / cfg.image_size[1])
    h_count = math.ceil(test_image[0].shape[0] / cfg.image_size[0])

    plt.imsave(OUTPUT_DIR / cfg.exp_name / "test_image.png", test_image[0])

    tile_array = []
    stack_pred = np.vstack(test_preds).reshape(-1, cfg.image_size[0], cfg.image_size[1])
    for h_i in range(h_count):
        tile_array.append(stack_pred[h_i * w_count : (h_i + 1) * w_count])

    pred_tile_image = concat_tile(tile_array)
    plt.imsave(OUTPUT_DIR / cfg.exp_name / "test_pred_tile_image.png", pred_tile_image)

    pred_tile_image = np.where(
        test_mask > 1,
        pred_tile_image[: test_image[0].shape[0], : test_image[0].shape[1]],
        0,
    )
    return pred_tile_image


def test(cfg: CFG, threshold: float = 0.4) -> pd.DataFrame:
    test_data_root = DATA_DIR / "test"
    preds = []
    for fp in test_data_root.glob("*"):
        print(fp)
        pred_tile_image = predict(cfg=cfg, test_data_dir=fp)

        if IS_TRAIN:
            plt.imsave(
                OUTPUT_DIR
                / cfg.exp_name
                / f"test_pred_tile_image_{str(fp).replace('/', '_')}.png",
                pred_tile_image,
            )
            plt.imsave(
                OUTPUT_DIR
                / cfg.exp_name
                / f"test_pred_tile_image_mask_{str(fp).replace('/', '_')}.png",
                np.where(pred_tile_image > threshold, 1, 0),
            )
        else:
            plt.imshow(pred_tile_image)
            plt.title("pred_tile_image")
            plt.show()
            plt.imshow(np.where(pred_tile_image > threshold, 1, 0))
            plt.title("pred_tile_mask")
            plt.show()

        starts_idx, lengths = rle(pred_tile_image, threshold=threshold)
        inklabels_rle = " ".join(map(str, sum(zip(starts_idx, lengths), ())))
        preds.append({"Id": str(fp).split("/")[-1], "Predicted": inklabels_rle})
    return pd.DataFrame(preds)


# =======================================================================
# main part
# =======================================================================
def main() -> None:
    cfg = CFG()
    (OUTPUT_DIR / cfg.exp_name).mkdir(parents=True, exist_ok=True)

    if IS_TRAIN:
        images = get_surface_volume_images()
        image_labels = get_inklabels_images()
        image_masks = get_mask_images()
        data_fold = split_cv(images=images, labels=image_labels, masks=image_masks)
        train(cfg=cfg, images=images, labels=image_labels, masks=image_masks)
        valid(cfg, data_fold=data_fold)
    if MAKE_SUB:
        preds = test(cfg, threshold=0.4)
        print(preds)
        save_path = (
            "submission.csv"
            if not IS_TRAIN
            else OUTPUT_DIR / cfg.exp_name / "submission.csv"
        )
        preds.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
