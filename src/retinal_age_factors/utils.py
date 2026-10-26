"""Identifier normalization, value parsing, and atomic file writes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_mrn(value: object) -> str | None:
    """Return an MRN as a zero-padded seven-digit string (join key form)."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    else:
        try:
            number = float(text)
            if math.isfinite(number) and number.is_integer():
                text = str(int(number))
        except (TypeError, ValueError):
            pass
    return text.zfill(7) if text.isdigit() else text


def mrn_string(series: pd.Series) -> pd.Series:
    """Return MRNs as plain strings (``1234567.0`` becomes ``1234567``)."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def normalize_date(value: object) -> str | None:
    """Return a date as ``YYYYMMDD`` for chart/image joins."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{8})(?:\.0+)?", text)
    if match:
        return match.group(1)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y%m%d")


def normalize_zip(value: object) -> str | None:
    """Return a ZIP or ZIP+4 value as its five-digit ZIP/ZCTA key."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{5})(?:-\d{4})?(?:\.0+)?$", text)
    if match:
        return match.group(1)
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            digits = str(int(number))
            return digits.zfill(5) if digits.isdigit() else None
    except (TypeError, ValueError):
        pass
    return None


def numeric_values(series: pd.Series) -> pd.Series:
    """Parse numbers, including values such as ``<0.1``, without imputing."""
    direct = pd.to_numeric(series, errors="coerce").astype(float)
    text = series.astype("string").str.replace(",", "", regex=False)
    extracted = pd.to_numeric(text.str.extract(r"([-+]?\d*\.?\d+)", expand=False), errors="coerce")
    return direct.fillna(extracted.astype(float))


def parse_image_dates(series: pd.Series) -> pd.Series:
    raw = series.astype("string")
    ymd = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    return ymd.combine_first(pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed"))


def parse_clinical_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), errors="coerce", dayfirst=True, format="mixed")


def first_nonempty(values: pd.Series) -> object:
    values = values.dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.iloc[0] if len(values) else np.nan


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)
