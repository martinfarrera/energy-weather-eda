# Energy and Weather Exploratory Analysis

Reproducible exploratory data analysis of household appliance energy use and weather conditions using the [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset.

## Project questions

1. How does household appliance energy use vary by time of day, and what observational relationships appear between consumption and indoor/outdoor conditions?
2. How accurately can completed household and weather observations predict total appliance energy in the next complete hour?

The exploratory analysis describes associations only. The forecast is predictive rather than causal.

## Repository layout

- `src/energy_analysis.py`: loading, validation, time features, summaries, correlations, and deterministic data exports.
- `src/energy_report.py`: static plots and exploratory Markdown report generation.
- `src/energy_forecast.py`: hourly feature construction, chronological model comparison, deterministic exports, and forecast report generation.
- `data/raw/`: downloaded source data and provenance record.
- `reports/energy-weather-eda.md`: generated exploratory report.
- `reports/energy-forecast.md`: generated one-hour-ahead forecasting report.
- `reports/*.png`: generated visualizations.
- `reports/data/`: generated summary tables, forecast metrics, and test predictions.
- `tests/`: public behavior tests.

## Reproduce

The project expects Python 3.10+ and uses pandas, NumPy, Matplotlib, scikit-learn, and pytest.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -c "from src.energy_analysis import load_energy_data, write_analysis_outputs; from src.energy_report import build_report; f=load_energy_data('data/raw/energydata_complete.csv'); write_analysis_outputs(f, 'reports/data'); build_report(f, 'reports')"
python -m src.energy_forecast --data data/raw/energydata_complete.csv --output-dir reports
```

Dataset download and licensing details are documented in [`data/raw/README.md`](data/raw/README.md).

## Main findings

- Mean appliance consumption is lowest around 03:00 and highest around 18:00 in the observed period.
- The Pearson correlation with outdoor temperature is weak (`r = 0.099`), while the correlation with lighting use is higher (`r = 0.197`).
- Indoor temperature variables are strongly correlated with one another, which matters for predictive modeling because of multicollinearity.
- The forecast compares persistence, standardized ridge regression, and a fixed-seed random forest. Model selection uses validation MAE; the final comparison uses an untouched chronological test partition.

The forecast aggregates six ten-minute readings into complete hours and uses only the completed forecast-origin hour and earlier appliance history. It predicts the following complete hour's total appliance energy. Source random columns `rv1` and `rv2` are excluded.

The data covers one Belgian household from January 11 through May 27, 2016. The limited household, geography, season, and observation window constrain generalization. The rolling evaluation also assumes completed-hour weather observations are available at prediction time; the source does not establish real-time publication latency.
