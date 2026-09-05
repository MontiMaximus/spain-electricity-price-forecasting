# Spanish day-ahead electricity price forecasting

Forecasting the 24 hourly clearing prices of the Spanish day-ahead electricity market
(OMIE, published through ENTSO-E): the auction closes at 12:00 CET on day D and clears
all 24 hours of day D+1 in a single shot, so this is not a rolling one-step-ahead problem
— the whole next-day profile has to be predicted at once, from information available today.

<!-- TODO: screenshot of the Streamlit app -->
<!-- TODO: live demo link -->

## Results

| Model                     | MAE (€/MWh) | MASE |
|---------------------------|-------------|------|
| ARIMA (non-seasonal)      | 37.21       | 1.89 |
| Seasonal Naive (weekly)   | 18.76       | 0.95 |
| Seasonal Naive (daily)    | 15.50       | 0.79 |
| SARIMA (24h seasonality)  | 14.46       | 0.74 |
| **XGBoost**               | **12.58**   | **0.64** |

MAE is averaged over the 10 delivery days, read from `data/processed/cv_metrics.csv`. The
delivery day is the unit of evaluation because the auction clears 24 hours at once. Every
window holds exactly 24 hours, so pooling all 240 gives the identical MAE — the choice of
aggregation only moves RMSE, which is not linear in the errors. RMSE keeps the same
ranking: 45.77, 24.11, 20.74, 19.36 and 16.52 in table order, averaged the same way.

**Cross-validation protocol.** Horizon `h = 24`, one full delivery day per window, 10
non-overlapping windows (2026-08-05 to 2026-08-14). Every window trains only on data
strictly before its delivery day. MASE divides a model's MAE by the MAE of a seasonal-naive
(24h) forecast over the history strictly before the first test day — that denominator, not
a package default, is what the MASE column is scaled by.

**Not an equal-data comparison.** XGBoost is refit on the full history at every fold;
SARIMA sees only the last 60 days and, with `refit=False`, has its orders and coefficients
estimated once and held fixed across the windows, because fitting AutoARIMA on four years
is computationally impractical. The table compares realistic deployments of each model
class, not the classes at equal data.

**One seasonal regime.** The 10 test days are consecutive days in August 2026 — high solar,
high cooling demand. These results do not generalise to winter.

## Why MAE and not MAPE

Spanish prices go negative. 4.4% of all hours since October 2022 clear below zero, 12.8%
of 2026 so far, and over 30% in the worst month (May 2025), with zero itself appearing
regularly. MAPE divides by the actual value, so it is undefined at zero and explodes near
it; it would also reward a model for being wrong on cheap hours. MAE is in EUR/MWh, which
is the unit the error actually costs, and MASE puts that on a scale where 1.0 means "no
better than copying yesterday".

## Leakage audit

- **Every price-derived feature is shifted by at least 24 hours.** The auction clears
  hour 23 of D+1 at the same moment as hour 00, so a lag shorter than 24h would use a
  price that does not exist yet at gate closure. `lag_24h`, `lag_48h`, `lag_168h` and the
  7-day rolling mean are all built on `shift(24)` or longer.
- **Splits are chronological, never random.** Every CV window trains strictly before its
  delivery day. A shuffled split on a time series leaks the future into the past.
- **Nothing is tuned against the test windows.** The booster runs a fixed 600 trees with
  no early stopping, so there is no stopping rule that could peek at the delivery day; if
  early stopping is added later it has to be scored on a validation slice carved out of
  the training window, never on the test day. Test data is touched once, to report.
- The hourly grid is made explicit with `asfreq("h")` before any shift, because `.shift()`
  counts positions, not clock time: a missing timestamp would silently misalign every lag
  after it.

## Where the model underperforms

XGBoost does not convincingly beat SARIMA yet. It reports a lower MAE, but the two are not
trained on the same amount of history (60 days versus the full four years), the backtest is
only 10 delivery days, and the confidence interval around a difference that size is wider
than the difference itself.

The working hypothesis is that without exogenous inputs — wind, solar and demand forecasts,
which are the physical drivers of Spanish prices — a tree model on calendar features and
own-price lags can only approximate a smoothed seasonal naive. Feature importance is
consistent with this: `lag_24h` alone carries roughly half the gain. Adding ENTSO-E
generation and load forecasts is the next step, and it is the change most likely to
produce a real improvement rather than a nicer-looking number.

## Limitations

- **DST handling.** Notebook 01 converts the UTC index to `Europe/Madrid` and then drops
  the timezone, so the CSV is naive Madrid local time. The October changeover produces a
  duplicated 02:00, which is dropped (4 rows across the dataset); the March changeover
  produces a gap, which `.asfreq("h")` fills with NaN (4 rows: 2023-03-26, 2024-03-31,
  2025-03-30 and 2026-03-29, all at 02:00). Because shifts count positions on that
  explicit grid, lag alignment holds either way, but the affected days carry 23 or 25 real
  hours and are not flagged as such.
- **10 backtest days is a small sample**, chosen against AutoARIMA's fitting cost.
- **No exogenous variables.** See the section above.
- **The dataset is frozen** at 2026-08-14 on purpose, so that training and evaluation stay
  reproducible. The app reads that frozen snapshot; there is no live ENTSO-E call yet.

## Repo map

```
notebooks/01_spain_hourly_dataset.ipynb   ENTSO-E extraction, parsing, cleaning
notebooks/02_EDA.ipynb                    exploratory analysis
notebooks/03_baseline_models.ipynb        Naive / Seasonal Naive / ARIMA / SARIMA
notebooks/04_xgboost.ipynb                features, cross-validation, final model
src/paths.py                              project paths, so nothing is relative
src/features.py                           feature engineering (mirrors notebook 04)
src/predict.py                            D+1 inference
app/app.py                                Streamlit dashboard over the frozen artifacts
data/raw/                                 untouched ENTSO-E XML (not committed)
data/processed/                           hourly price CSV, CV predictions, CV metrics
models/                                   the fitted XGBoost booster
```

The notebooks are self-contained: they define their feature code inline instead of
importing `src/`. The duplication is deliberate — a notebook should read top to bottom
without jumping to another file — and `src/features.py` carries a note saying that a
change to a feature has to be made in both places.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .                 # makes `from src... import ...` work everywhere

copy .env.example .env           # then paste your ENTSO-E token into it
streamlit run app/app.py
```

Only notebook 01 needs the API token; everything else reads `data/processed/`.

## Data source

[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), day-ahead prices
(`documentType=A44`) for the Spanish bidding zone (`10YES-REE------0`), from 2022-10-01.

Spain moved to 15-minute market time units in October 2025, so the series has two eras:
hourly (`PT60M`) until 2025-09-30 and quarter-hourly (`PT15M`) afterwards. Notebook 01
mean-aggregates the four quarters of each hour, which reproduces the official 60-minute
index, so the target stays "the hourly day-ahead price" across the whole period.
