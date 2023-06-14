from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import ttach as tta
from albumentations.pytorch import ToTensorV2
from loguru import logger
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import TypeAlias
from warmup_scheduler import GradualWarmupScheduler

import wandb
from src.augmentations import cutmix, label_noise, mixup
from src.losses.soft_dice_loss import SoftDiceLossV2


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
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: LossFn,
        adv_param: str = "weight",
        adv_lr: float = 0.2,
        adv_eps: int = 1,
        start_epoch: int = 0,
        adv_step: int = 1,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
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
            self._attack_step()
            with autocast(device_type="cuda", enabled=self.scaler is not None):
                logits = self.model(x)["ink"]
                adv_loss = self.criterion(logits, y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

        self._restore()

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
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


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(False)


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


def freeze_model(model: nn.Module, freeze_keys: list[str] = ["encoder"]) -> None:
    """freeze model parameters specified by freeze_keys

    Args:
        model (nn.Module): model
    """
    for name, param in model.named_parameters():
        contains = [key in name for key in freeze_keys]
        if any(contains):
            param.requires_grad = False


def resize_image_with_half_size(image: torch.Tensor) -> torch.Tensor:
    """resize image

    Args:
        image (torch.Tensor): image, (B, H, W)

    Returns:
        torch.Tensor: resized image
    """
    if image.ndim == 3:
        raise ValueError(f"image must be (B, H, W), but got {image.shape}")
    H, W = image.shape[-2:]
    image = tv.transforms.Resize((H // 2, W // 2))(image)
    return image


def random_depth_crop(
    volume: torch.Tensor, depth_range: tuple[int, int], crop_depth: int
) -> tuple[torch.Tensor, int, int]:
    """random depth crop

    Args:
        volume (torch.Tensor): volume, (C, D, H, W)
        depth_range (tuple[int, int]): depth range
        crop_depth (int): crop depth
    """
    z0 = int(np.random.randint(low=0, high=volume.shape[1] - crop_depth, size=1))
    z1 = z0 + crop_depth
    # logger.debug(f"volume.shape: {volume.shape} z0: {z0}, z1: {z1}")
    # assert z1 <= volume.shape[1]
    return volume[:, z0:z1, :, :], z0, z1


LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train_per_epoch(
    cfg,
    model: nn.Module,
    train_loader: DataLoader,
    criterion: LossFn,
    scaler: torch.cuda.amp.GradScaler,
    fold: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    criterion_cls: Optional[LossFn] = None,
    use_awp: bool = False,
    schedule_per_step: bool = False,
) -> float:
    # Map of Config
    mixup_alpha = cfg.mixup_alpha
    cutmix_alpha = cfg.cutmix_alpha

    start_epoch_to_freeze_model = cfg.start_freaze_model_epoch
    is_frozen = False
    freeze_keys = cfg.freeze_keys
    max_grad_norm = cfg.max_grad_norm
    use_amp = cfg.use_amp

    depth_range = cfg.fragment_z
    fragment_depth = cfg.fragment_depth

    if use_awp:
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
    with tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
        desc="Train Per Epoch",
    ) as pbar:
        for step, (images, target) in pbar:
            model.train()

            if cfg.cutmix and np.random.rand() <= cfg.cutmix_prob:
                images, target, _, _ = mixup(images, target, alpha=mixup_alpha)

            if cfg.mixup and np.random.rand() <= cfg.mixup_prob:
                images, target, _, _ = cutmix(images, target, alpha=cutmix_alpha)

            if cfg.label_noise and np.random.rand() <= cfg.label_noise_prob:
                images, target, _ = label_noise(images, target)

            images = images.contiguous().to(cfg.device, non_blocking=True)
            target = target.contiguous().to(cfg.device, non_blocking=True)
            batch_size = target.size(0)
            target_cls = make_cls_label(target)

            images, _, _ = random_depth_crop(images, depth_range, fragment_depth)
            # logger.debug(images.shape)

            if not is_frozen and epoch > start_epoch_to_freeze_model:
                logger.info(f"freeze model with {freeze_keys}")
                freeze_model(model, freeze_keys=freeze_keys)
                is_frozen = True

            # target1 = resize_image_with_half_size(target)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                logit1 = outputs["logit1"]
                logit2 = outputs["logit2"]
                loss1 = criterion(logit1, target)
                loss2 = criterion(logit2, target)
                loss_mask = loss1 + loss2

                if any(["cls" in out_key for out_key in outputs.keys()]) and criterion_cls is not None:
                    weight_cls = cfg.weight_cls
                    # pred_label = outputs["pred_label_logits"]
                    cls_logits1 = outputs["cls_logits1"]
                    cls_logits2 = outputs["cls_logits2"]
                    loss_cls1 = weight_cls * criterion_cls(cls_logits1, target_cls)
                    loss_cls2 = weight_cls * criterion_cls(cls_logits2, target_cls)
                    loss_cls = loss_cls1 + loss_cls2
                else:
                    loss_cls = 0

                loss = loss_mask + loss_cls
                loss /= cfg.grad_accum

            running_loss.update(value=loss.item(), n=batch_size)
            scaler.scale(loss).backward()
            # multiple lossの時は別々に勾配計算？
            # Ref:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
            # scaler.scale(loss_mask).backward(retain_graph=True)
            # scaler.scale(loss_cls).backward()

            if use_awp:
                awp.attack_backward(images, target, step)

            if (step + 1) % cfg.grad_accum == 0:
                # unscale -> clip
                # Ref
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                # scaler.unscale_(optimizer)
                # clip gradient of parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if schedule_per_step:
                    scheduler.step()

                pbar.set_postfix(
                    {
                        "fold": f"{fold}",
                        "epoch": f"{epoch}",
                        "loss_avg": f"{running_loss.avg:.4f}",
                        "cls_loss": f"{loss_cls.item():.4f}",
                        "loss": f"{loss.item():.4f}",
                    }
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        f"fold{fold}_train_loss": loss.item(),
                        "learning_rate": learning_rate,
                        f"fold{fold}_cls_train_loss": loss_cls.item(),
                    }
                )

    return running_loss.avg


def tta_rotate(net: nn.Module, volume: torch.Tensor) -> torch.Tensor:
    B, _, H, W = volume.shape
    rotated_volumes = [
        volume,
        torch.rot90(volume, k=1, dims=(-2, -1)),
        torch.rot90(volume, k=2, dims=(-2, -1)),
        torch.rot90(volume, k=3, dims=(-2, -1)),
    ]
    K = len(rotated_volumes)
    # batch = {
    #     "volume": torch.cat(rotated_volumes, dim=0),
    # }
    volume = torch.cat(rotated_volumes, dim=0)
    output = net(volume)
    ink = output["ink"]
    ink = ink.reshape(K, B, 1, H, W)
    ink = [
        ink[0],
        torch.rot90(ink[1], k=-1, dims=(-2, -1)),
        torch.rot90(ink[2], k=-2, dims=(-2, -1)),
        torch.rot90(ink[3], k=-3, dims=(-2, -1)),
    ]
    ink = torch.stack(ink, dim=0).mean(dim=0)
    return ink


def valid_per_epoch(
    cfg,
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: Callable,
    fold: int,
    epoch: int,
    valid_xyxys: np.ndarray,
    valid_masks: np.ndarray,
    log_prefix: str = "",
) -> dict[str, float | np.ndarray]:
    crop_size = cfg.crop_size
    depth_range = cfg.fragment_z
    fragment_depth = cfg.fragment_depth

    mask_preds = np.zeros(valid_masks.shape)
    mask_count = np.zeros(valid_masks.shape)
    model.eval()
    valid_losses = AverageMeter(name="valid_loss")
    valid_bces = AverageMeter(name="valid_bce")

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

        image, _, _ = random_depth_crop(image, depth_range, fragment_depth)

        # batch = {"volume": image}
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                # segm_logits: (N, 1, H, W)
                if cfg.use_tta:
                    y_preds = tta_rotate(model, image)
                else:
                    y_preds = model(image)["ink"]
            loss_mask = criterion(y_preds, target)

            # cls: (N, 1)
            # pred = model(image)
            # pred_logtis = pred["pred_label_logits"]
            # loss_cls = nn.BCEWithLogitsLoss()(pred_logtis, target_cls)
            # accs = ((pred_logtis > 0.5) == target_cls).sum().item() / batch_size
            # loss = loss_mask + (cfg.weight_cls * loss_cls)
            loss = loss_mask

        bce = F.binary_cross_entropy_with_logits(input=y_preds, target=target)
        valid_losses.update(value=loss.item(), n=batch_size)
        valid_bces.update(value=bce.item(), n=batch_size)
        wandb.log(
            {
                f"{log_prefix}fold{fold}_valid_loss": loss.item(),
                f"{log_prefix}fold{fold}_valid_bce": bce.item(),
                # f"fold{fold}_valid_cls_loss": loss_cls.item(),
                # f"fold{fold}_valid_acc": accs,
            }
        )

        # make a whole image prediction
        # y_preds: (N, H, W)
        y_preds = y_preds.to("cpu").detach().numpy()
        assert y_preds.shape == (batch_size, 1, crop_size, crop_size)
        start_idx = step * batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_preds[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((crop_size, crop_size))

    logger.info(
        f"mask_count_min: {mask_count.min()}, mask_count_max: {mask_count.max()}, zero_sum: {(mask_count == 0).sum()}"
    )
    mask_preds /= mask_count + 1e-7
    return {
        "valid_loss_avg": valid_losses.avg,
        "mask_preds": mask_preds,
        "valid_bce_avg": valid_bces.avg,
    }


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


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
                + f" for fold {self.fold} with best score {self.best_score}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.logger_fn(f"Detected Increasing Score: best score {self.best_score} --> {score}")
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, prefix: str = "") -> None:
        """Save model when validation loss decrease."""
        if self.verbose:
            self.logger_fn(f"Validation loss decreased ({self.val_loss_min} --> {val_loss})")

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


# =========================================================
# Factories
# =========================================================
def get_scheduler(
    cfg, optimizer: nn.optim.Optimizer, step_per_epoch: int | None = None
) -> optim.lr_scheduler._LRScheduler:
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=1e-6)
        return scheduler
    if cfg.scheduler == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            # epochs=cfg.epoch,
            # steps_per_epoch=step_per_epoch,
            total_steps=step_per_epoch,
            max_lr=cfg.max_lr,
            # pct_start=0.1,
            anneal_strategy="cos",
            div_factor=1e1,
            final_div_factor=1e2,
        )
        return scheduler
    if cfg.scheduler == "TwoCyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=cfg.encoder_lr,
            max_lr=cfg.max_lr,
        )
        return scheduler
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
        # total_epoch: warmupでtarget_lrの値に到達するまでのepoch数
        # multiplier: target_lr = multiplier * base_lr
        # after_scheduler: warmup後に適用するscheduler
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.T_max, eta_min=1e-7)
        scheduler = GradualWarmupSchedulerV2(
            optimizer=optimizer,
            multiplier=10,
            total_epoch=1,
            after_scheduler=scheduler_cosine,
        )
        return scheduler

    raise ValueError(f"Invalid scheduler: {cfg.scheduler}")


def get_loss(cfg) -> LossFn:
    if cfg.loss == "BCEWithLogitsLoss":
        loss = nn.BCEWithLogitsLoss()
        return loss

    if cfg.loss == "SoftBCE":
        loss = smp.losses.SoftBCEWithLogitsLoss()
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
            return alpha * bce_loss(y_pred, y_true) + (1 - alpha) * dice_loss(y_pred, y_true)

        return _loss
    if cfg.loss == "BCELovaszLoss":

        def _loss(y_pred, y_true, alpha=0.5):
            bce_loss = nn.BCEWithLogitsLoss()
            lobasz_loss = smp.losses.LovaszLoss(mode="binary", from_logits=True)
            return alpha * bce_loss(y_pred, y_true) + (1 - alpha) * lobasz_loss(y_pred, y_true)

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
        loss = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.7, beta=0.3)
        return loss

    if cfg.loss == "BCETverskyLoss":

        def _loss(y_pred, y_true, lamb=0.5):
            bce_loss = nn.BCEWithLogitsLoss()
            tversky_loss = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.7, beta=0.3)
            return lamb * bce_loss(y_pred, y_true) + (1 - lamb) * tversky_loss(y_pred, y_true)

        return _loss

    if cfg.loss == "BCEDiceTverskyLoss":

        def _loss(y_pred, y_true):
            bce_loss = nn.BCEWithLogitsLoss()
            dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
            tversky_loss = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.7, beta=0.3)
            return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true) + 2 * tversky_loss(y_pred, y_true)

        return _loss

    if cfg.loss == "BCESoftDiceLoss":

        def _loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor, bce_weight: float = cfg.bce_weight) -> torch.Tensor:
            bce_loss = nn.BCEWithLogitsLoss()
            dice_loss = SoftDiceLossV2()

            if math.isclose(bce_weight, 1.0):
                return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)

            return bce_weight * bce_loss(y_pred, y_true) + (1 - bce_weight) * dice_loss(y_pred, y_true)

        return _loss_fn

    raise ValueError(f"Invalid loss: {cfg.loss}")


def get_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    if cfg.use_diff_lr:
        params = [
            {
                "params": [param for name, param in model.named_parameters() if "encoder" in name],
                "lr": cfg.encoder_lr,
            },
            {
                "params": [param for name, param in model.named_parameters() if "decoder" in name],
                "lr": cfg.decoder_lr,
            },
        ]
    else:
        params = model.parameters()
    if cfg.optimizer == "AdamW":
        weight_decay = getattr(cfg, "weight_decay")
        if cfg.use_diff_lr:
            if weight_decay is not None:
                optimizer = optim.AdamW(params=params, weight_decay=weight_decay)
            else:
                optimizer = optim.AdamW(params=params)
        else:
            if weight_decay is not None:
                optimizer = optim.AdamW(params=params, lr=cfg.lr, weight_decay=weight_decay)
            else:
                optimizer = optim.AdamW(params=params, lr=cfg.lr)
        return optimizer
    if cfg.optimizer == "RAdam":
        if cfg.use_diff_lr:
            optimizer = optim.RAdam(params=params, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.RAdam(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        return optimizer
    raise ValueError(f"{cfg.optimizer} is not supported")


def get_alb_transforms(phase: str, cfg) -> A.Compose | tuple[A.Compose, A.Compose]:
    """
    Args:
        phase: {"train", "valid", "test"}
        cfg: 設定
    """
    image_size = (cfg.crop_size, cfg.crop_size)
    in_chans = cfg.fragment_z[1] - cfg.fragment_z[0]
    if phase == "train":
        return A.Compose(cfg.train_compose), A.Compose(cfg.soft_train_compose)
    elif phase == "valid":
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif phase == "test":
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        raise ValueError(f"Invalid phase: {phase}")
