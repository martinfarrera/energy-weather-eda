# Raw dataset provenance

## Source

- Dataset: UCI Appliances Energy Prediction (ID 374)
- Repository page: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- Download archive: https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Download date: 2026-10-13
- Local source file: `energydata_complete.csv`
- SHA-256: `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`

## Reproducible download

```bash
curl -L --fail --silent --show-error \
  'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip' \
  -o uci-appliances-energy.zip
unzip -o uci-appliances-energy.zip
```

Verify the extracted CSV with the SHA-256 recorded above before analysis.

## Shape and coverage

- Rows: 19,735
- Columns: 29
- Time range: 2016-01-11 17:00:00 through 2016-05-27 18:00:00
- Sampling interval: 10 minutes
- Duplicate rows: 0
- Missing values in the downloaded CSV: 0

## Schema

- `date`: timestamp at ten-minute resolution.
- `Appliances`: appliance energy use in Wh; primary outcome.
- `lights`: lighting energy use in Wh.
- `T1`–`T9`: indoor temperatures in Celsius across household rooms.
- `RH_1`–`RH_9`: indoor relative humidity percentages across household rooms.
- `T_out`: outdoor temperature in Celsius.
- `Press_mm_hg`: atmospheric pressure in mm Hg.
- `RH_out`: outdoor relative humidity percentage.
- `Windspeed`: wind speed in m/s.
- `Visibility`: visibility in km.
- `Tdewpoint`: outdoor dew point in Celsius.
- `rv1`, `rv2`: random variables included by the source dataset.

## Attribution

The dataset was created by Luis M. Candanedo, Véronique Feldheim, and David Deramaix. The weather observations were obtained from the nearest airport weather station at Chievres Airport, Belgium, and the UCI page documents permission for distributing the weather data.

This project does not claim that weather variables cause household energy use. Findings are observational associations within this dataset and period.
