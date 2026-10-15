"""Deterministic one-hour-ahead energy forecasting pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .energy_analysis import load_energy_data, validate_energy_data

TARGET_COLUMN = "target_next_hour_wh"
FEATURE_COLUMNS = [
    "appliances_lag_1_wh",
    "appliances_lag_2_wh",
    "appliances_trailing_3h_mean_wh",
    "lights_last_hour_wh",
    "T_out_last_hour_mean",
    "RH_out_last_hour_mean",
    "Windspeed_last_hour_mean",
    "Visibility_last_hour_mean",
    "Tdewpoint_last_hour_mean",
    "target_hour_sin",
    "target_hour_cos",
    "target_day_of_week",
    "target_is_weekend",
]


def _validate_forecast_timestamps(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("forecast data must have a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("forecast data must use a unique, continuous ten-minute cadence")
    if len(frame) < 2 or not (frame.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all():
        raise ValueError("forecast data must use a unique, continuous ten-minute cadence")


def build_hourly_forecast_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Create past-only features for total appliance energy in the next hour."""
    validate_energy_data(frame)
    _validate_forecast_timestamps(frame)

    aggregations = {
        "Appliances": "sum",
        "lights": "sum",
        "T_out": "mean",
        "RH_out": "mean",
        "Windspeed": "mean",
        "Visibility": "mean",
        "Tdewpoint": "mean",
    }
    hourly = frame[list(aggregations)].resample("h").agg(aggregations)
    counts = frame["Appliances"].resample("h").count()
    hourly = hourly.loc[counts == 6].copy()

    # A gap or partial bucket would make adjacent rows look consecutive after filtering.
    if len(hourly) < 5 or not (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("forecast data must contain consecutive complete hours")

    target_time = hourly.index.to_series().shift(-1)
    target_hour = target_time.dt.hour
    dataset = pd.DataFrame(index=hourly.index)
    dataset.index.name = "forecast_origin"
    dataset["target_time"] = target_time
    dataset["appliances_lag_1_wh"] = hourly["Appliances"]
    dataset["appliances_lag_2_wh"] = hourly["Appliances"].shift(1)
    dataset["appliances_trailing_3h_mean_wh"] = hourly["Appliances"].rolling(3).mean()
    dataset["lights_last_hour_wh"] = hourly["lights"]
    for column in ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]:
        dataset[f"{column}_last_hour_mean"] = hourly[column]
    dataset["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    dataset["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    dataset["target_day_of_week"] = target_time.dt.dayofweek
    dataset["target_is_weekend"] = (target_time.dt.dayofweek >= 5).astype(float)
    dataset[TARGET_COLUMN] = hourly["Appliances"].shift(-1)
    return dataset.dropna().copy()


def chronological_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an ordered forecast dataset without shuffling or label overlap."""
    if dataset.empty or not dataset.index.is_monotonic_increasing:
        raise ValueError("forecast dataset must be non-empty and chronologically ordered")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave non-empty train, validation, and test partitions")

    train_end = int(len(dataset) * train_fraction)
    validation_end = int(len(dataset) * (train_fraction + validation_fraction))
    train = dataset.iloc[:train_end].copy()
    validation = dataset.iloc[train_end:validation_end].copy()
    test = dataset.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("forecast dataset is too short for the requested chronological split")
    if train["target_time"].max() > validation.index.min():
        raise ValueError("training labels overlap the validation forecast period")
    if validation["target_time"].max() > test.index.min():
        raise ValueError("validation labels overlap the test forecast period")
    return train, validation, test


def _models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
    }


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_wh": float(mean_absolute_error(actual, predicted)),
        "rmse_wh": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def _write_report(results: dict[str, Any], output_dir: Path) -> Path:
    metrics = results["test_metrics"]
    selected = results["selected_model"]
    baseline_mae = metrics["persistence"]["mae_wh"]
    selected_mae = metrics[selected]["mae_wh"]
    improvement = (baseline_mae - selected_mae) / baseline_mae * 100
    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {values['mae_wh']:.2f} | {values['rmse_wh']:.2f} | {values['r2']:.3f} |"
        for name, values in metrics.items()
    )
    report = f"""# One-hour-ahead Household Energy Forecast

## Question

How accurately can completed household and weather observations predict total appliance energy in the next complete hour?

## Method

Ten-minute readings are aggregated into complete hourly totals or means. Features use only the completed forecast-origin hour and earlier appliance history. The target is the following complete hour's appliance total. Random source columns `rv1` and `rv2` are excluded.

The evaluation uses a chronological 60/20/20 train, validation, and test split. Ridge and random forest models are selected by validation MAE, then refit on train plus validation data. Persistence—predicting that the next hour equals the last completed hour—is the baseline. The final test partition is untouched during model selection.

## Test results

| Model | MAE (Wh) | RMSE (Wh) | R² |
| --- | ---: | ---: | ---: |
{rows}

The validation-selected model is **{selected.replace('_', ' ')}**. Its test MAE is {selected_mae:.2f} Wh, a {improvement:.1f}% change relative to persistence (positive means improvement).

## Interpretation and limitations

This is a rolling one-hour-ahead evaluation: each forecast may use observations from the hour that has just completed. Weather measurements are treated as available after that hour, but their original publication timing is not established here.

Results describe one Belgian household from January through May 2016. They do not establish performance for other households, climates, seasons, or deployment conditions. The analysis is predictive rather than causal.

## Reproduction

```bash
python -m pytest -q
python -m src.energy_forecast --data data/raw/energydata_complete.csv --output-dir reports
```
"""
    report_path = output_dir / "energy-forecast.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_forecast(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Fit candidate models, evaluate chronologically, and write deterministic artifacts."""
    dataset = build_hourly_forecast_dataset(frame)
    train, validation, test = chronological_split(dataset)
    models = _models()

    validation_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        validation_metrics[name] = _metrics(
            validation[TARGET_COLUMN],
            model.predict(validation[FEATURE_COLUMNS]),
        )
    selected_model = min(validation_metrics, key=lambda name: validation_metrics[name]["mae_wh"])

    fitting = pd.concat([train, validation])
    test_predictions: dict[str, np.ndarray] = {
        "persistence": test["appliances_lag_1_wh"].to_numpy(),
    }
    for name, model in _models().items():
        model.fit(fitting[FEATURE_COLUMNS], fitting[TARGET_COLUMN])
        test_predictions[name] = model.predict(test[FEATURE_COLUMNS])

    test_metrics = {
        name: _metrics(test[TARGET_COLUMN], prediction)
        for name, prediction in test_predictions.items()
    }
    results: dict[str, Any] = {
        "target": "total appliance energy in the next complete hour (Wh)",
        "selected_model": selected_model,
        "selection_metric": "validation MAE (Wh)",
        "features": FEATURE_COLUMNS,
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "validation_start": str(validation.index.min()),
            "validation_end": str(validation.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameters": {
            "ridge": {"alpha": 10.0, "standardized": True},
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 4,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_path = Path(output_dir)
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "forecast_metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "forecast_origin": test.index,
            "target_time": test["target_time"].to_numpy(),
            "actual_next_hour_wh": test[TARGET_COLUMN].to_numpy(),
            **{f"predicted_{name}_wh": value for name, value in test_predictions.items()},
        }
    )
    predictions.to_csv(data_path / "forecast_predictions.csv", index=False, float_format="%.6f")
    _write_report(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic one-hour-ahead energy forecasting.")
    parser.add_argument("--data", required=True, help="Path to the UCI energy CSV.")
    parser.add_argument("--output-dir", default="reports", help="Directory for report artifacts.")
    args = parser.parse_args()
    frame = load_energy_data(args.data)
    results = run_forecast(frame, args.output_dir)
    print(json.dumps({"selected_model": results["selected_model"], "test_metrics": results["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
