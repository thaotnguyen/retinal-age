"""AI-READI external validation of a model trained on all CUTI images.

Stages: ``prepare-aireadi`` converts AI-READI color fundus DICOMs to PNG with
AutoMorph's fundus crop; ``run-automorph`` applies AutoMorph's 8-model image
quality ensemble (same filter as CUTI); ``train-final`` fits one model on all
CUTI images for the median best epoch across CV folds; ``external`` predicts
every good-quality AI-READI image and reports patient-level metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import config
from ..utils import atomic_csv, write_json
from .cv import MANIFEST_DIR, summarize_cv
from .retfound import CHECKPOINT_REPO, checkpoint_sha256
from .training import TrainConfig, load_trained_model, patient_predictions, predict_images, regression_metrics, train_model

LOG = logging.getLogger(__name__)
EXTERNAL_DIR = config.MODEL_DIR / "aireadi"
QUALITY_MANIFEST = EXTERNAL_DIR / "aireadi_quality_manifest.csv"
FINAL_DIR = config.MODEL_DIR / "checkpoints" / "final"
AUTOMORPH_SEEDS = [(7, 28), (6, 30), (5, 32), (4, 34), (3, 36), (2, 38), (1, 40), (0, 42)]


def _import_from(directory: Path, module: str):
    sys.path.insert(0, str(directory))
    try:
        return __import__(module)
    finally:
        sys.path.pop(0)


def _build_manifest() -> pd.DataFrame:
    participants = pd.read_csv(config.AIREADI_DIR / "participants.tsv", sep="\t", dtype={"participant_id": str})
    retinal = pd.read_csv(config.AIREADI_DIR / "retinal_photography" / "manifest.tsv", sep="\t", dtype={"participant_id": str})
    retinal = retinal[retinal["filepath"].astype(str).str.contains("/cfp/")]
    keep = ["participant_id", "age", "clinical_site", "study_group", "study_visit_date"]
    manifest = retinal.merge(participants[keep], on="participant_id", how="left", validate="many_to_one")
    if manifest["age"].isna().any():
        raise ValueError("AI-READI CFP rows are missing participant age")
    manifest["source_path"] = manifest["filepath"].map(lambda p: str(config.AIREADI_DIR / str(p).lstrip("/")))
    manifest["image_id"] = [
        "aireadi_" + hashlib.sha256(f"{p}\0{uid}".encode()).hexdigest()[:24]
        for p, uid in zip(manifest["filepath"], manifest["sop_instance_uid"])
    ]
    manifest["path"] = manifest["image_id"] + ".png"
    manifest["quality_status"] = "not_run"
    manifest["quality_error"] = ""
    return manifest


def _dicom_to_rgb(path: Path) -> Image.Image:
    dataset = pydicom.dcmread(path)
    array = np.asarray(dataset.pixel_array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    array = array[..., :3]
    if array.dtype != np.uint8:
        lo, hi = np.percentile(array, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(array.min()), float(array.max())
        array = (np.clip((array.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = 255 - array
    return Image.fromarray(array)


def prepare_aireadi() -> None:
    manifest = pd.read_csv(QUALITY_MANIFEST, dtype={"participant_id": str}) if QUALITY_MANIFEST.exists() else _build_manifest()
    manifest["quality_error"] = manifest["quality_error"].fillna("").astype(str)
    image_dir = EXTERNAL_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fundus_prep = _import_from(config.AUTOMORPH_DIR / "M0_Preprocess", "fundus_prep")

    def prepare(index: int) -> tuple[int, str, str]:
        destination = image_dir / manifest.at[index, "path"]
        if destination.is_file():
            return index, "prepared", ""
        try:
            rgb = np.asarray(_dicom_to_rgb(Path(manifest.at[index, "source_path"])))
            cropped, *_ = fundus_prep.process_without_gb(rgb.copy(), rgb.copy(), [], [], [])
            Image.fromarray(np.asarray(cropped).astype(np.uint8)).save(destination, format="PNG")
            return index, "prepared", ""
        except Exception as exc:  # an unreadable DICOM is recorded, not fatal
            return index, "unreadable", repr(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, future in enumerate(as_completed([pool.submit(prepare, i) for i in manifest.index]), start=1):
            index, status, error = future.result()
            manifest.loc[index, ["quality_status", "quality_error"]] = [status, error]
            if done % 500 == 0:
                atomic_csv(manifest, QUALITY_MANIFEST)
                LOG.info("Prepared %d/%d images", done, len(manifest))
    atomic_csv(manifest, QUALITY_MANIFEST)
    LOG.info("AI-READI preparation: %s", manifest["quality_status"].value_counts().to_dict())


class _QualityDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as image:
            array = np.asarray(image.convert("RGB").resize((512, 512)), dtype=np.float32)
        nonzero = array[array > 0]
        mean = float(nonzero.mean()) if nonzero.size else 0.0
        std = float(nonzero.std()) if nonzero.size else 1.0
        return torch.from_numpy(((array - mean) / max(std, 1e-6)).transpose(2, 0, 1))


def run_automorph(devices: list[str]) -> None:
    """AutoMorph M1 quality ensemble; 'good' = predicted good, or usable with
    P(bad) < 0.01 (the AutoMorph good-quality rule used for CUTI)."""
    manifest = pd.read_csv(QUALITY_MANIFEST, dtype={"participant_id": str}).fillna({"quality_error": ""})
    todo = manifest[manifest["quality_status"].isin(["prepared", "good", "bad"])]
    module_dir = config.AUTOMORPH_DIR / "M1_Retinal_Image_quality_EyePACS"
    efficientnet = _import_from(module_dir, "model").Efficientnet_fl
    models: list[tuple[nn.Module, torch.device]] = []
    for i, (split, seed) in enumerate(AUTOMORPH_SEEDS):
        device = torch.device(devices[i % len(devices)])
        model = efficientnet(pretrained=False)
        weights = module_dir / "Retinal_quality" / "EyePACS_quality" / "efficientnet" / f"{split}_seed_{seed}" / "best_loss_checkpoint.pth"
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        models.append((model.to(device).eval(), device))

    loader = DataLoader(_QualityDataset([EXTERNAL_DIR / "images" / p for p in todo["path"]]), batch_size=16, num_workers=8)
    probabilities = []
    with torch.inference_mode():
        for images in loader:
            batch = torch.stack([torch.softmax(m(images.to(d)), dim=1).cpu() for m, d in models]).mean(0)
            probabilities.append(batch.numpy())
    probs = np.concatenate(probabilities)
    predicted = probs.argmax(axis=1)
    good = (predicted == 0) | ((predicted == 1) & (probs[:, 2] < 0.01))
    manifest.loc[todo.index, ["softmax_good", "softmax_usable", "softmax_bad"]] = probs
    manifest.loc[todo.index, "quality_status"] = np.where(good, "good", "bad")
    atomic_csv(manifest, QUALITY_MANIFEST)
    LOG.info("AI-READI quality: %s", manifest["quality_status"].value_counts().to_dict())


def train_final(device: str = "cuda:0") -> None:
    """Train on all CUTI images for the median CV best epoch."""
    best_epochs = [
        json.loads((config.MODEL_DIR / "checkpoints" / f"fold_{k}" / "training_summary.json").read_text())["best_epoch"]
        for k in range(config.N_FOLDS)
    ]
    epochs = max(1, int(round(float(np.median(best_epochs)))))
    train_config = TrainConfig()
    rows = pd.read_csv(MANIFEST_DIR / "cuti_images.csv", dtype={"MRN": str})
    train_model(rows, None, config.IMAGE_DIR, FINAL_DIR, train_config, torch.device(device), stop_after=epochs)
    write_json(
        {
            "cuti_oof": summarize_cv(),
            "fold_best_epochs": best_epochs,
            "final_epochs": epochs,
            "checkpoint_repo": CHECKPOINT_REPO,
            "checkpoint_sha256": checkpoint_sha256(),
        },
        config.MODEL_DIR / "metrics" / "final_model.json",
    )


def external(device: str = "cuda:0") -> None:
    manifest = pd.read_csv(QUALITY_MANIFEST, dtype={"participant_id": str})
    good = manifest[manifest["quality_status"] == "good"].rename(columns={"participant_id": "MRN", "study_visit_date": "date"})
    if good.empty:
        raise RuntimeError("No AI-READI images passed AutoMorph")
    train_config = TrainConfig()
    model = load_trained_model(FINAL_DIR / "best.pt", train_config, torch.device(device))
    images = predict_images(model, good, EXTERNAL_DIR / "images", train_config, torch.device(device))
    patients = patient_predictions(images)
    sites = good.groupby("MRN")["clinical_site"].first()
    patients["clinical_site"] = patients["patient_id"].map(sites)
    atomic_csv(images, EXTERNAL_DIR / "aireadi_image_predictions.csv")
    atomic_csv(patients, EXTERNAL_DIR / "aireadi_patient_predictions.csv")
    write_json(
        {
            "overall": regression_metrics(patients["true_age"], patients["predicted_age"]),
            "by_clinical_site": {
                str(site): regression_metrics(g["true_age"], g["predicted_age"]) for site, g in patients.groupby("clinical_site")
            },
            "n_participants_without_good_images": int(manifest["participant_id"].nunique() - len(patients)),
        },
        config.MODEL_DIR / "metrics" / "aireadi_external.json",
    )
