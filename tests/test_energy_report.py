from pathlib import Path

from src.energy_analysis import load_energy_data
from src.energy_report import build_report


DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "energydata_complete.csv"


def test_build_report_creates_plots_and_reproducible_markdown(tmp_path):
    frame = load_energy_data(DATA_PATH)
    report_path = build_report(frame, tmp_path)

    assert report_path == tmp_path / "energy-weather-eda.md"
    assert report_path.exists()
    assert (tmp_path / "hourly_consumption.png").stat().st_size > 0
    assert (tmp_path / "consumption_weather.png").stat().st_size > 0
    report = report_path.read_text()
    assert "## Findings" in report
    assert "observational" in report.lower()
    assert "hourly_consumption.png" in report
