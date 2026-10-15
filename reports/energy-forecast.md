# One-hour-ahead Household Energy Forecast

## Question

How accurately can completed household and weather observations predict total appliance energy in the next complete hour?

## Method

Ten-minute readings are aggregated into complete hourly totals or means. Features use only the completed forecast-origin hour and earlier appliance history. The target is the following complete hour's appliance total. Random source columns `rv1` and `rv2` are excluded.

The evaluation uses a chronological 60/20/20 train, validation, and test split. Ridge and random forest models are selected by validation MAE, then refit on train plus validation data. Persistence—predicting that the next hour equals the last completed hour—is the baseline. The final test partition is untouched during model selection.

## Test results

| Model | MAE (Wh) | RMSE (Wh) | R² |
| --- | ---: | ---: | ---: |
| Persistence | 198.83 | 372.23 | 0.225 |
| Ridge | 197.25 | 323.61 | 0.414 |
| Random Forest | 185.17 | 311.95 | 0.456 |

The validation-selected model is **random forest**. Its test MAE is 185.17 Wh, a 6.9% change relative to persistence (positive means improvement).

## Interpretation and limitations

This is a rolling one-hour-ahead evaluation: each forecast may use observations from the hour that has just completed. Weather measurements are treated as available after that hour, but their original publication timing is not established here.

Results describe one Belgian household from January through May 2016. They do not establish performance for other households, climates, seasons, or deployment conditions. The analysis is predictive rather than causal.

## Reproduction

```bash
python -m pytest -q
python -m src.energy_forecast --data data/raw/energydata_complete.csv --output-dir reports
```
