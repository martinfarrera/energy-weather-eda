from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.energy_analysis import load_energy_data
from src.energy_forecast import (
    FEATURE_COLUMNS,
    build_hourly_forecast_dataset,
    chronological_split,
    run_forecast,
)


DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "energydata_complete.csv"


def make_ten_minute_frame(hours: int = 8) -> pd.DataFrame:
    index = pd.date_range("2016-01-01", periods=hours * 6, freq="10min", name="date")
    values = np.arange(1, len(index) + 1, dtype=float)
    return pd.DataFrame(
        {
            "Appliances": values,
            "lights": np.ones(len(index)),
            "T_out": np.linspace(5, 12, len(index)),
            "RH_out": np.linspace(80, 60, len(index)),
            "Windspeed": np.full(len(index), 3.0),
            "Visibility": np.full(len(index), 40.0),
            "Tdewpoint": np.linspace(2, 4, len(index)),
            "rv1": values[::-1],
            "rv2": values[::-1],
        },
        index=index,
    )


def test_build_hourly_forecast_dataset_aligns_past_features_with_next_hour_total():
    frame = make_ten_minute_frame()

    dataset = build_hourly_forecast_dataset(frame)

    first = dataset.iloc[0]
    assert first.name == pd.Timestamp("2016-01-01 02:00:00")
    assert first["target_time"] == pd.Timestamp("2016-01-01 03:00:00")
    assert first["appliances_lag_1_wh"] == pytest.approx(sum(range(13, 19)))
    assert first["appliances_lag_2_wh"] == pytest.approx(sum(range(7, 13)))
    assert first["appliances_trailing_3h_mean_wh"] == pytest.approx(
        np.mean([sum(range(1, 7)), sum(range(7, 13)), sum(range(13, 19))])
    )
    assert first["target_next_hour_wh"] == pytest.approx(sum(range(19, 25)))
    assert "rv1" not in FEATURE_COLUMNS
    assert "rv2" not in FEATURE_COLUMNS


def test_build_hourly_forecast_dataset_rejects_irregular_timestamps():
    frame = make_ten_minute_frame().drop(pd.Timestamp("2016-01-01 01:20:00"))

    with pytest.raises(ValueError, match="continuous ten-minute cadence"):
        build_hourly_forecast_dataset(frame)


def test_chronological_split_keeps_order_and_labels_before_next_partition():
    dataset = build_hourly_forecast_dataset(make_ten_minute_frame(hours=30))

    train, validation, test = chronological_split(dataset)

    assert len(train) + len(validation) + len(test) == len(dataset)
    assert train.index.max() < validation.index.min() < test.index.min()
    assert train["target_time"].max() <= validation.index.min()
    assert validation["target_time"].max() <= test.index.min()


def test_run_forecast_writes_deterministic_metrics_predictions_and_report(tmp_path):
    frame = load_energy_data(DATA_PATH)

    first = run_forecast(frame, tmp_path)
    first_metrics = (tmp_path / "data" / "forecast_metrics.json").read_bytes()
    first_predictions = (tmp_path / "data" / "forecast_predictions.csv").read_bytes()
    second = run_forecast(frame, tmp_path)

    assert first["selected_model"] in {"ridge", "random_forest"}
    assert first["target"] == "total appliance energy in the next complete hour (Wh)"
    assert first["split"]["train_rows"] > first["split"]["validation_rows"] > 0
    assert first["split"]["test_rows"] > 0
    assert set(first["test_metrics"]) == {"persistence", "ridge", "random_forest"}
    assert first_metrics == (tmp_path / "data" / "forecast_metrics.json").read_bytes()
    assert first_predictions == (tmp_path / "data" / "forecast_predictions.csv").read_bytes()
    assert first == second

    report = (tmp_path / "energy-forecast.md").read_text()
    assert "one-hour-ahead" in report
    assert "chronological" in report.lower()
    assert "one Belgian household" in report
