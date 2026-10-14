# Energy and Weather Exploratory Analysis

Reproducible exploratory data analysis of household appliance energy use and weather conditions using the [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset.

## Project question

How does household appliance energy use vary by time of day, and what observational relationships appear between consumption and indoor/outdoor conditions?

This project describes associations only. It does not claim that weather variables cause energy consumption.

## Repository layout

- `src/energy_analysis.py`: loading, validation, time features, summaries, correlations, and deterministic data exports.
- `src/energy_report.py`: static plots and Markdown report generation.
- `data/raw/`: downloaded source data and provenance record.
- `reports/energy-weather-eda.md`: generated narrative report.
- `reports/*.png`: generated visualizations.
- `reports/data/`: generated summary tables.
- `tests/`: public behavior tests.

## Reproduce

The project expects Python 3.10+ and uses pandas, NumPy, Matplotlib, and pytest.

```bash
python -m pip install pandas numpy matplotlib pytest
python -m pytest -q
python -c "from src.energy_analysis import load_energy_data, write_analysis_outputs; from src.energy_report import build_report; f=load_energy_data('data/raw/energydata_complete.csv'); write_analysis_outputs(f, 'reports/data'); build_report(f, 'reports')"
```

Dataset download and licensing details are documented in [`data/raw/README.md`](data/raw/README.md).

## Main findings

- Mean appliance consumption is lowest around 03:00 and highest around 18:00 in the observed period.
- The Pearson correlation with outdoor temperature is weak (`r = 0.099`), while the correlation with lighting use is higher (`r = 0.197`).
- Indoor temperature variables are strongly correlated with one another, which matters for any future predictive modeling because of multicollinearity.

The data covers one Belgian household from January 11 through May 27, 2016, sampled every ten minutes. The limited household, geography, season, and observation window constrain generalization.
