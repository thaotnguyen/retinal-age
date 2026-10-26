"""Patient-exclusive 10-fold cross-validation on the CUTI cohort.

Folds come from ``StratifiedGroupKFold`` over image rows, grouped by MRN and
stratified by age decile (seed 2026), so no patient appears in more than one
fold.  For outer fold k, fold k is the test set, fold k+1 (mod 10) the
validation set, and the remaining eight folds train (80/10/10).  The split
assignment depends on scikit-learn's implementation; it reproduces exactly
with scikit-learn 1.7.2 (pinned in ``pyproject.toml``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .. import config
from ..utils import atomic_csv, write_json
from .training import TrainConfig, patient_predictions, predict_images, regression_metrics, train_model

LOG = logging.getLogger(__name__)
MANIFEST_DIR = config.MODEL_DIR / "manifests"


def make_splits() -> None:
    images = pd.read_csv(config.IMAGES_CSV)
    data = images[["path", "MRN", "date", "age", "sex"]].copy()
    data["MRN"] = data["MRN"].astype(str)
    # Stable, nearly equal-count age deciles even when ages are tied.
    data["age_decile"] = pd.qcut(data["age"].rank(method="first"), q=10, labels=False).astype(int)
    data["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SPLIT_SEED)
    for fold, (_, test_idx) in enumerate(splitter.split(data, data["age_decile"], groups=data["MRN"])):
        data.loc[test_idx, "fold"] = fold
    if (data["fold"] < 0).any() or (data.groupby("MRN")["fold"].nunique() > 1).any():
        raise AssertionError("Fold assignment is incomplete or not patient-exclusive")

    atomic_csv(data, MANIFEST_DIR / "cuti_images.csv")
    for fold in range(config.N_FOLDS):
        split = pd.Series("train", index=data.index)
        split[data["fold"] == fold] = "test"
        split[data["fold"] == (fold + 1) % config.N_FOLDS] = "val"
        atomic_csv(data.assign(split=split), MANIFEST_DIR / f"cuti_fold_{fold}.csv")
    LOG.info("Wrote manifests for %d images / %d patients", len(data), data["MRN"].nunique())


def train_fold(fold: int, device: str = "cuda:0") -> None:
    """Train one outer fold and predict its held-out patients."""
    train_config = TrainConfig()
    manifest = pd.read_csv(MANIFEST_DIR / f"cuti_fold_{fold}.csv", dtype={"MRN": str})
    rows = {name: manifest[manifest["split"] == name] for name in ("train", "val", "test")}
    model = train_model(
        rows["train"], rows["val"], config.IMAGE_DIR, config.MODEL_DIR / "checkpoints" / f"fold_{fold}",
        train_config, torch.device(device),
    )  # fmt: skip
    images = predict_images(model, rows["test"], config.IMAGE_DIR, train_config, torch.device(device))
    patients = patient_predictions(images).assign(fold=fold)
    atomic_csv(images, config.MODEL_DIR / "predictions" / f"cuti_fold_{fold}_images.csv")
    atomic_csv(patients, config.MODEL_DIR / "predictions" / f"cuti_fold_{fold}_patients.csv")
    write_json(
        regression_metrics(patients["true_age"], patients["predicted_age"]),
        config.MODEL_DIR / "metrics" / f"cuti_fold_{fold}.json",
    )


def run_cv(gpus: list[int]) -> None:
    """Run the ten folds as subprocesses, one per GPU at a time."""
    pending = list(range(config.N_FOLDS))
    running: list[tuple[int, int, subprocess.Popen, object]] = []
    log_dir = config.MODEL_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        busy = {gpu for _, gpu, _, _ in running}
        for gpu in [g for g in gpus if g not in busy]:
            if not pending:
                break
            fold = pending.pop(0)
            handle = (log_dir / f"fold_{fold}.log").open("w")
            command = [sys.executable, "-m", "retinal_age_factors", "train-fold", "--fold", str(fold)]
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
            running.append((fold, gpu, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
            LOG.info("Started fold %d on GPU %d", fold, gpu)
        time.sleep(5)
        for item in list(running):
            fold, _, process, handle = item
            if process.poll() is None:
                continue
            handle.close()
            running.remove(item)
            if process.returncode != 0:
                raise RuntimeError(f"Fold {fold} failed; see {log_dir / f'fold_{fold}.log'}")
            LOG.info("Completed fold %d", fold)
    summarize_cv()


def summarize_cv() -> dict:
    """Pooled out-of-fold metrics at the patient level (one row per patient)."""
    patients = pd.concat(
        [
            pd.read_csv(config.MODEL_DIR / "predictions" / f"cuti_fold_{k}_patients.csv", dtype={"patient_id": str})
            for k in range(config.N_FOLDS)
        ],
        ignore_index=True,
    )
    if patients["patient_id"].duplicated().any():
        raise AssertionError("A patient appears in more than one held-out fold")
    summary = {
        "evaluation": "patient-exclusive 10-fold out-of-fold predictions, averaged per patient",
        "patient_metrics": regression_metrics(patients["true_age"], patients["predicted_age"]),
        "fold_patient_metrics": {
            str(k): regression_metrics(g["true_age"], g["predicted_age"]) for k, g in patients.groupby("fold")
        },
        "n_images": int(patients["n_images"].sum()),
        "n_patients": int(len(patients)),
    }
    atomic_csv(patients, config.MODEL_DIR / "cuti_oof_patient_predictions.csv")
    write_json(summary, config.MODEL_DIR / "metrics" / "cuti_oof_summary.json")
    return summary
