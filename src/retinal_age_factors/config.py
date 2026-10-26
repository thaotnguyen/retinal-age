"""Paths and fixed study constants.

Every input lives under ``DATA_DIR`` (default ``<repo>/data``; override with the
``RAF_DATA_DIR`` environment variable) and every generated file under
``OUTPUT_DIR`` (default ``<repo>/outputs``; override with ``RAF_OUTPUT_DIR``).
Neither directory is tracked by git.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RAF_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("RAF_OUTPUT_DIR", REPO_ROOT / "outputs"))
RESOURCES_DIR = REPO_ROOT / "resources"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"

# Restricted CUTI inputs (protected health information; never committed).
VISITS_CSV = DATA_DIR / "Good_quality_only.csv"          # quality-filtered screening visits
IMAGES_CSV = DATA_DIR / "human_data.csv"                 # one row per quality-filtered image
IMAGE_DIR = DATA_DIR / "Good_quality"                    # AutoMorph-passed fundus photographs
RDV_CSV = DATA_DIR / "RDV0002447_v6.csv"                 # EHR vitals, labs, and medications
RUCA_CSV = DATA_DIR / "Modified_RUCA_Zip_Codes_2010.csv"  # ZIP-level RUCA codes
AIREADI_DIR = DATA_DIR / "aireadi"                       # AI-READI dataset root (external validation)

# Public contextual data, downloaded by ``python -m retinal_age_factors download``.
PUBLIC_DIR = DATA_DIR / "public"

# Third-party code that is cloned rather than vendored.
AUTOMORPH_DIR = THIRD_PARTY_DIR / "AutoMorph"

# Generated outputs.
MODEL_DIR = OUTPUT_DIR / "retinal_age"
DATASET_CSV = OUTPUT_DIR / "full_rag_data.csv"
PATIENTS_CSV = OUTPUT_DIR / "full_rag_data_patients.csv"

# Study constants.
N_FOLDS = 10
SPLIT_SEED = 2026
CLINICAL_MATCH_DAYS = 183  # nearest EHR record within six months of the image
