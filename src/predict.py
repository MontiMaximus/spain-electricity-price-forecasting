"""Inference for the Spanish day-ahead price model.

Mirrors the forecasting cell of notebooks/04_xgboost.ipynb. The features are
rebuilt with the same create_features() used in training, so training and
serving cannot drift apart.
"""

import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET, create_features

HORIZON = 24    # the auction clears 24 hours at once

# Minimum history the feature chain needs before the first forecast hour.
# The longest chain is same_hour_mean_7d: a 7-step rolling window on y.shift(24).
# It reaches back to t-168, the same depth as lag_168h, so 168 is the computed
# minimum. 192 is a deliberate safety margin of one extra day.
MIN_HISTORY_HOURS = 192


def forecast_next_day(model, history_df: pd.DataFrame) -> pd.DataFrame:
    """Return 24 hourly predictions for D+1.

    `history_df` is the raw price frame (a TARGET column on an hourly
    DatetimeIndex), not a feature matrix. The last day it contains is day D;
    the 24 hours forecast are those of D+1.
    """
    if len(history_df) < MIN_HISTORY_HOURS:
        raise ValueError(
            f"forecast_next_day needs at least {MIN_HISTORY_HOURS} hours of history "
            f"(168 to build complete features, plus a day of margin), got "
            f"{len(history_df)}.")

    target_day   = history_df.index.max().normalize() + pd.Timedelta(days=1)
    future_index = pd.date_range(target_day, periods=HORIZON, freq="h")

    # empty rows for the target day. Calendar features come from the timestamp,
    # price features from the concatenated history.
    future   = pd.DataFrame({TARGET: np.nan}, index=future_index)
    combined = pd.concat([history_df[[TARGET]], future])

    X = create_features(combined).loc[future_index]

    # dropna on FEATURES only, never on TARGET: TARGET is NaN for all 24 rows by
    # construction (that is what we are predicting), so dropping on it would
    # delete the whole forecast window. A row surviving here with a NaN feature
    # would be a hole in the history, and the frame comes back shorter than 24.
    X = X.dropna(subset=FEATURES)

    # XGBoost matches features by POSITION, not by name: predicting on columns in
    # a different order does not raise, it silently returns wrong numbers. So the
    # order is taken from the fitted booster itself rather than from FEATURES.
    X = X[model.get_booster().feature_names]

    return pd.DataFrame({"forecast_eur_mwh": model.predict(X)}, index=X.index)
