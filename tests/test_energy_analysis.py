from pathlib import Path

import pandas as pd
import pytest

from src.energy_analysis import (
    add_time_features,
    build_summary,
    compute_correlations,
    load_energy_data,
    summarize_by_hour,
    validate_energy_data,
    write_analysis_outputs,
)


DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "energydata_complete.csv"


def test_load_energy_data_parses_timestamp_and_numeric_columns():
    frame = load_energy_data(DATA_PATH)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert frame.index.name == "date"
    assert frame["Appliances"].dtype.kind in "fi"
    assert frame.shape == (19_735, 28)


def test_validate_energy_data_rejects_missing_required_columns():
    frame = pd.DataFrame({"date": ["2016-01-01"], "Appliances": [1]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_energy_data(frame)


def test_add_time_features_exposes_calendar_and_cyclical_fields():
    frame = load_energy_data(DATA_PATH).iloc[:1]
    enriched = add_time_features(frame)

    assert enriched.loc[enriched.index[0], "hour"] == 17
    assert enriched.loc[enriched.index[0], "day_of_week"] == 0
    assert "hour_sin" in enriched
    assert "month" in enriched


def test_build_summary_reports_quality_and_consumption_statistics():
    frame = load_energy_data(DATA_PATH)
    summary = build_summary(frame)

    assert summary["rows"] == 19_735
    assert summary["columns"] == 28
    assert summary["missing_cells"] == 0
    assert summary["duplicate_rows"] == 0
    assert summary["appliances_mean_wh"] == pytest.approx(97.69, abs=0.1)
    assert summary["appliances_max_wh"] == pytest.approx(1080.0)


def test_summarize_by_hour_covers_all_hours_and_mean_consumption():
    frame = load_energy_data(DATA_PATH)
    hourly = summarize_by_hour(frame)

    assert list(hourly.columns) == ["hour", "mean_appliances_wh", "median_appliances_wh", "observations"]
    assert hourly["hour"].tolist() == list(range(24))
    assert hourly["observations"].sum() == len(frame)
    assert hourly["mean_appliances_wh"].max() > hourly["mean_appliances_wh"].min()


def test_compute_correlations_returns_named_target_correlations():
    frame = load_energy_data(DATA_PATH)
    correlations = compute_correlations(frame)

    assert correlations.index[0] == "Appliances"
    assert correlations.columns[0] == "Appliances"
    assert correlations.loc["Appliances", "Appliances"] == pytest.approx(1.0)
    assert correlations.shape == (28, 28)


def test_write_analysis_outputs_creates_reproducible_csv_and_json_files(tmp_path):
    frame = load_energy_data(DATA_PATH)
    outputs = write_analysis_outputs(frame, tmp_path)

    assert sorted(path.name for path in outputs) == [
        "correlations.csv",
        "hourly_summary.csv",
        "summary.json",
    ]
    assert (tmp_path / "hourly_summary.csv").read_text().startswith("hour,")
    assert '"rows": 19735' in (tmp_path / "summary.json").read_text()
