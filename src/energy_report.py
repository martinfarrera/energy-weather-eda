"""Generate a compact static visual report for the energy-weather EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .energy_analysis import build_summary, compute_correlations, summarize_by_hour


def build_report(frame, output_dir: str | Path) -> Path:
    """Write plots and a Markdown report from an already validated frame."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hourly = summarize_by_hour(frame)
    correlations = compute_correlations(frame)
    summary = build_summary(frame)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["mean_appliances_wh"], marker="o", color="#1769aa")
    ax.set(title="Average appliance energy use by hour", xlabel="Hour of day", ylabel="Appliances (Wh)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path / "hourly_consumption.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sample = frame.iloc[::10]
    ax.scatter(sample["T_out"], sample["Appliances"], s=8, alpha=0.25, color="#c2410c")
    ax.set(title="Outdoor temperature and appliance energy use", xlabel="Outdoor temperature (°C)", ylabel="Appliances (Wh)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path / "consumption_weather.png", dpi=160)
    plt.close(fig)

    peak = hourly.loc[hourly["mean_appliances_wh"].idxmax()]
    trough = hourly.loc[hourly["mean_appliances_wh"].idxmin()]
    weather_corr = correlations.loc["Appliances", "T_out"]
    lights_corr = correlations.loc["Appliances", "lights"]
    report = f"""# Energy and Weather Exploratory Analysis

Dataset: UCI Appliances Energy Prediction (ID 374). The observation period runs from {summary['start']} to {summary['end']} at ten-minute intervals.

## Quality overview

- {summary['rows']:,} rows and {summary['columns']} numeric analysis columns after parsing `date` as the index.
- {summary['missing_cells']} missing cells and {summary['duplicate_rows']} duplicate rows in the downloaded CSV.
- Mean appliance consumption: {summary['appliances_mean_wh']:.2f} Wh; median: {summary['appliances_median_wh']:.2f} Wh; maximum: {summary['appliances_max_wh']:.0f} Wh.

## Findings

- Mean appliance consumption is lowest around {int(trough['hour']):02d}:00 ({trough['mean_appliances_wh']:.1f} Wh) and highest around {int(peak['hour']):02d}:00 ({peak['mean_appliances_wh']:.1f} Wh).
- The Pearson correlation between appliance consumption and outdoor temperature is {weather_corr:.3f}; the correlation with lighting use is {lights_corr:.3f}.
- Indoor temperature variables are strongly correlated with one another, so they should not be interpreted as independent signals without addressing multicollinearity.

These are observational associations in one Belgian household over roughly four and a half months. They do not establish that weather or any other variable causes appliance consumption, and the random variables `rv1` and `rv2` are retained only as source columns, not interpreted as meaningful drivers.

## Visuals

![Average hourly consumption](hourly_consumption.png)

![Outdoor temperature and appliance consumption](consumption_weather.png)

## Reproduction

```bash
python -m pytest -q
python -c "from src.energy_analysis import load_energy_data, write_analysis_outputs; from src.energy_report import build_report; f=load_energy_data('data/raw/energydata_complete.csv'); write_analysis_outputs(f, 'reports/data'); build_report(f, 'reports')"
```
"""
    report_path = output_path / "energy-weather-eda.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
