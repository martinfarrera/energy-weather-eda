"""Reusable loading and exploratory summaries for the energy-weather dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Appliances",
    "lights",
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
}


def load_energy_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV with a validated, chronological datetime index."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("date").set_index("date")
    validate_energy_data(frame)
    return frame


def validate_energy_data(frame: pd.DataFrame) -> None:
    """Validate the minimum public data contract for analysis."""
    columns = set(frame.columns)
    has_date = "date" in columns or isinstance(frame.index, pd.DatetimeIndex)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if not has_date:
        missing.append("date")
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) and "date" not in frame.columns:
        raise ValueError("data must have a datetime index or a date column")
    if frame.empty:
        raise ValueError("data must contain at least one row")
    if frame["Appliances"].isna().any():
        raise ValueError("Appliances contains missing values")
    if (frame["Appliances"] < 0).any():
        raise ValueError("Appliances cannot be negative")


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar and cyclical time features."""
    result = frame.copy()
    index = result.index if isinstance(result.index, pd.DatetimeIndex) else pd.to_datetime(result["date"])
    result["hour"] = index.hour
    result["day_of_week"] = index.dayofweek
    result["month"] = index.month
    result["is_weekend"] = (index.dayofweek >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    return result


def summarize_by_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize appliance consumption by hour of day."""
    validate_energy_data(frame)
    working = add_time_features(frame)
    result = (
        working.groupby("hour", as_index=False)["Appliances"]
        .agg(
            mean_appliances_wh="mean",
            median_appliances_wh="median",
            observations="size",
        )
        .sort_values("hour")
    )
    result["observations"] = result["observations"].astype(int)
    return result


def compute_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlations for numeric analysis columns."""
    validate_energy_data(frame)
    return frame.select_dtypes(include="number").corr().loc[lambda data: data.index, :]


def write_analysis_outputs(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write deterministic summary tables used by the report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    hourly_path = output_path / "hourly_summary.csv"
    correlations_path = output_path / "correlations.csv"

    summary_path.write_text(
        json.dumps(build_summary(frame), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summarize_by_hour(frame).to_csv(hourly_path, index=False, float_format="%.6f")
    compute_correlations(frame).to_csv(correlations_path, float_format="%.6f")
    return [summary_path, hourly_path, correlations_path]


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Build compact quality and primary-outcome summary statistics."""
    validate_energy_data(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.reset_index().duplicated().sum()),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "appliances_mean_wh": float(frame["Appliances"].mean()),
        "appliances_median_wh": float(frame["Appliances"].median()),
        "appliances_max_wh": float(frame["Appliances"].max()),
        "lights_mean_wh": float(frame["lights"].mean()),
    }
