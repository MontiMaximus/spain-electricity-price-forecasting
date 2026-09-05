"""Feature engineering for the Spanish day-ahead price model.

This module mirrors code that is defined inline in notebooks/04_xgboost.ipynb.
The notebook is self-contained on purpose: it must read top-to-bottom without
jumping to other files. The duplication is therefore intentional, but it means
that if a feature changes it has to be changed in BOTH places.
"""

import numpy as np
import pandas as pd

# Spanish national holidays behave like Sundays in the demand profile.
# If the package is missing the notebook still runs, with the flag at 0.
try:
    import holidays as holidays_pkg
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("`holidays` not installed -> is_holiday will be all zeros. Run: pip install holidays")

# Defined in the notebook's data-loading cell; a module constant here so that
# create_price_features() keeps the same default argument it has in the notebook.
TARGET = "price_eur_mwh"


def create_calendar_features(data):

    """Features from the timestamp alone. These are the only ones knowable in advance,
    which is what lets us build a design matrix for a day that has not happened yet."""

    d = data.copy()

    d["hour"]        = d.index.hour
    d["day_of_week"] = d.index.dayofweek          # 0 = Monday
    d["month"]       = d.index.month
    d["is_weekend"]  = (d.index.dayofweek >= 5).astype(int)

    if HOLIDAYS_AVAILABLE:
        es = holidays_pkg.country_holidays(
            "ES", years=range(d.index.year.min(), d.index.year.max() + 1))
        d["is_holiday"] = pd.Series(d.index.date, index=d.index).isin(set(es)).astype(int)
    else:
        d["is_holiday"] = 0

    # circular encodings: keep 23h next to 00h, and December next to January
    d["hour_sin"]  = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"]  = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    return d


def create_price_features(data, target=TARGET):
    d = data.copy()
    y = d[target]

    d["lag_24h"]  = y.shift(24)      # same hour, day D
    d["lag_48h"]  = y.shift(48)      # same hour, day D-1
    d["lag_168h"] = y.shift(168)     # same hour, last week

    # mean of THIS hour over the last 7 days. Grouping by hour makes each rolling
    # step one day instead of one hour.
    d["same_hour_mean_7d"] = (y.shift(24).groupby(d.index.hour)
                                  .transform(lambda s: s.rolling(7, min_periods=7).mean()))
    return d


CALENDAR_FEATURES = ["hour", "hour_sin", "hour_cos",
                     "day_of_week", "is_weekend", "is_holiday",
                     "month", "month_sin", "month_cos"]
LAG_FEATURES      = ["lag_24h", "lag_48h", "lag_168h"]
ROLLING_FEATURES  = ["same_hour_mean_7d"]

FEATURES = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES


def create_features(data):
    """Raw price frame -> frame with all features.

    One function for training, cross-validation and live forecasting. A transformation
    that exists in only one of those paths is a training/serving skew bug.

    Expects `data` to be indexed by a DatetimeIndex on a complete hourly grid: the
    shifts count positions, not clock time, so a missing timestamp misaligns every
    lag after it.

    The longest chain is the 7-day window sitting on top of the 24h shift. It
    reaches back to t-168, the same depth as lag_168h, so the first 168 rows come
    out NaN and cannot be used. (src/predict.py guards with 192: a deliberate
    safety margin, not the computed minimum.)

    No dropna happens here on purpose: at inference the 24 target rows have an
    unknown price, and dropping on TARGET would delete exactly the rows to predict.
    """
    return create_price_features(create_calendar_features(data))
