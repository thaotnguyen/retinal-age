"""Fine-tuning recipe, image dataset, training loop, and prediction.

The recipe follows RETFound's official DINOv2 fine-tuning defaults
(https://github.com/rmaphoh/RETFound/blob/main/train.sh): full fine-tuning,
AdamW with layer-wise decay, warmup plus cosine schedule, RandAugment and
random erasing.  The only task change is a single-output head trained with
L1 loss on chronological age in years (the paper's MAE objective).
"""

from __future__ import annotations

import json
import logging
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..utils import write_json
from .retfound import build_model, param_groups_lrd

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    image_size: int = 224
    batch_size: int = 24
    epochs: int = 50
    warmup_epochs: int = 10
    base_lr: float = 5e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    layer_decay: float = 0.65
    drop_path: float = 0.2
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    num_workers: int = 8

    @property
    def learning_rate(self) -> float:
        return self.base_lr * self.batch_size / 256.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # as in RETFound's main_finetune.py


def is_left_eye(row: pd.Series) -> bool:
    """AI-READI rows carry ``laterality`` (L/R); CUTI filenames encode OD/OS."""
    if "laterality" in row.index:
        return str(row["laterality"]).upper().startswith("L")
    name = str(row["path"]).upper()
    return "-OS_" in name or "_OS_" in name


class RetinaAgeDataset(Dataset):
    """Fundus images with chronological age; left eyes are mirrored to match
    right-eye orientation (optic disc on the same side)."""

    def __init__(self, rows: pd.DataFrame, image_root: Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        with Image.open(self.image_root / str(row["path"])) as image:
            image = image.convert("RGB")
            if is_left_eye(row):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensor = self.transform(image)
        return tensor, torch.tensor(float(row["age"]), dtype=torch.float32), idx


def build_transforms(config: TrainConfig):
    train = create_transform(
        input_size=config.image_size,
        is_training=True,
        color_jitter=None,
        auto_augment=config.auto_augment,
        interpolation="bicubic",
        re_prob=config.reprob,
        re_mode=config.remode,
        re_count=config.recount,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    evaluate = transforms.Compose([
        transforms.Resize(int(config.image_size / (224 / 256)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return train, evaluate


def make_loader(dataset: Dataset, config: TrainConfig, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _set_lr(optimizer: torch.optim.Optimizer, progress: float, config: TrainConfig) -> None:
    if progress < config.warmup_epochs:
        lr = config.learning_rate * progress / config.warmup_epochs
    else:
        ratio = (progress - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        lr = config.min_lr + (config.learning_rate - config.min_lr) * 0.5 * (1 + math.cos(math.pi * ratio))
    for group in optimizer.param_groups:
        group["lr"] = lr * group["lr_scale"]


def predict_images(model: nn.Module, rows: pd.DataFrame, image_root: Path, config: TrainConfig, device: torch.device):
    """Return ``rows`` with a ``predicted_age`` column."""
    _, evaluate = build_transforms(config)
    loader = make_loader(RetinaAgeDataset(rows, image_root, evaluate), config, train=False)
    predicted = np.full(len(rows), np.nan)
    model.eval()
    with torch.inference_mode():
        for images, _, indices in loader:
            with _autocast(device):
                output = model(images.to(device, non_blocking=True)).flatten()
            predicted[indices.numpy()] = output.float().cpu().numpy()
    result = rows.reset_index(drop=True).copy()
    result["predicted_age"] = predicted
    return result


def patient_predictions(images: pd.DataFrame, patient_col: str = "MRN") -> pd.DataFrame:
    """Average image predictions (and ages) within each patient."""
    result = images.groupby(patient_col, as_index=False).agg(
        true_age=("age", "mean"),
        predicted_age=("predicted_age", "mean"),
        n_images=("predicted_age", "size"),
        n_visits=("date", "nunique"),
    )
    result = result.rename(columns={patient_col: "patient_id"})
    result["absolute_error"] = (result["predicted_age"] - result["true_age"]).abs()
    return result


def regression_metrics(true: pd.Series, predicted: pd.Series) -> dict:
    return {
        "n": int(len(true)),
        "mae": float(mean_absolute_error(true, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(true, predicted))),
        "r2": float(r2_score(true, predicted)),
        "pearson_r": float(pearsonr(true, predicted).statistic),
    }


def train_model(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame | None,
    image_root: Path,
    output_dir: Path,
    config: TrainConfig,
    device: torch.device,
    stop_after: int | None = None,
) -> nn.Module:
    """Train on the ``config.epochs`` schedule; keep the epoch with the lowest
    validation patient-level MAE (or the last epoch without a validation set).
    ``stop_after`` ends training early while keeping the same LR schedule."""
    set_seed(config.seed)
    train_transform, _ = build_transforms(config)
    loader = make_loader(RetinaAgeDataset(train_rows, image_root, train_transform), config, train=True)
    model = build_model(config.image_size, config.drop_path, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        param_groups_lrd(model, config.weight_decay, config.layer_decay), lr=config.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mae, best_epoch, history = float("inf"), 0, []

    for epoch in range(stop_after or config.epochs):
        model.train()
        total = 0.0
        for step, (images, targets, _) in enumerate(loader):
            _set_lr(optimizer, epoch + step / len(loader), config)
            with _autocast(device):
                pred = model(images.to(device, non_blocking=True)).flatten()
                loss = nn.functional.l1_loss(pred, targets.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        record = {"epoch": epoch + 1, "train_loss": total / len(loader)}
        improved = val_rows is None
        if val_rows is not None:
            val = patient_predictions(predict_images(model, val_rows, image_root, config, device))
            record["val_patient_mae"] = float(val["absolute_error"].mean())
            improved = record["val_patient_mae"] < best_mae
        if improved:
            best_mae, best_epoch = record.get("val_patient_mae", float("nan")), epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "config": asdict(config)}, output_dir / "best.pt")
        history.append(record)
        write_json(history, output_dir / "history.json")
        LOG.info(json.dumps(record))

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    write_json(
        {
            "best_epoch": best_epoch,
            "best_val_patient_mae": None if val_rows is None else best_mae,
            "config": asdict(config),
            "learning_rate": config.learning_rate,
            "n_train_images": int(len(train_rows)),
            "n_train_patients": int(train_rows["MRN"].nunique()),
            "n_validation_images": None if val_rows is None else int(len(val_rows)),
            "n_validation_patients": None if val_rows is None else int(val_rows["MRN"].nunique()),
        },
        output_dir / "training_summary.json",
    )
    return model


def load_trained_model(checkpoint: Path, config: TrainConfig, device: torch.device) -> nn.Module:
    model = build_model(config.image_size, config.drop_path, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return model.to(device).eval()
