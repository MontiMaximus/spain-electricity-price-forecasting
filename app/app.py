"""Streamlit dashboard for the Spanish day-ahead price forecasts.

Reads frozen artifacts only. It never trains and never calls the ENTSO-E API:
every number on screen was produced by notebooks 03 and 04 and written to
data/processed/. The only computation at runtime is a single XGBoost forward
pass in the Forecast tab.

Run with:  streamlit run app/app.py
"""

import altair as alt
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.paths import DATA_PROCESSED, MODELS_DIR
from src.features import TARGET
from src.predict import forecast_next_day

PRICES_CSV  = DATA_PROCESSED / "spain_day_ahead_hourly_2022_2026_madrid_time.csv"
PREDICTIONS = DATA_PROCESSED / "predictions.parquet"
CV_METRICS  = DATA_PROCESSED / "cv_metrics.csv"
MODEL_FILE  = MODELS_DIR / "xgboost_final.json"

GITHUB_URL = "TODO"      # TODO: point this at the public repo

st.set_page_config(page_title="Spanish day-ahead price forecasting", layout="wide")

# ----------------------------------------------------------------------------
# Series styling
# ----------------------------------------------------------------------------
# Defined once and keyed by series name so no chart can drift from another.
# Hierarchy: the observed price is the reference, XGBoost is the model this
# project is about, the baselines sit in the background. XGBoost and SARIMA are
# separated by blue against orange rather than by red against green, so the
# distinction survives the common forms of colour blindness; the baselines are
# desaturated and dashed so shape, not just hue, tells them apart.
ACTUAL = "Actual"

SERIES_STYLE = {
    ACTUAL:      {"color": "#333333", "width": 3.0, "dash": [1, 0]},
    "XGBoost":   {"color": "#0072B2", "width": 2.5, "dash": [1, 0]},
    "SARIMA":    {"color": "#C77700", "width": 1.8, "dash": [1, 0]},
    "SN Daily":  {"color": "#8A8A8A", "width": 1.2, "dash": [5, 2]},
    "SN Weekly": {"color": "#6B7C8C", "width": 1.2, "dash": [2, 2]},
    "ARIMA":     {"color": "#9C8264", "width": 1.2, "dash": [7, 3]},
}
FALLBACK_STYLE = {"color": "#767676", "width": 1.2, "dash": [3, 3]}

PRICE_AXIS = "€/MWh"
ERROR_AXIS = "MAE (€/MWh)"


def style(series):
    """Style for one series name, falling back for a model not listed above."""
    return SERIES_STYLE.get(series, FALLBACK_STYLE)


def series_order(metrics, present):
    """Legend order: ground truth first, then the model this project is about,
    then the baselines ranked by mean MAE. The ranking is read off cv_metrics.csv
    rather than hardcoded, so it stays correct when the numbers change."""
    ranked = metrics.groupby("model")["mae"].mean().sort_values().index.tolist()
    order  = [ACTUAL, "XGBoost"] + [m for m in ranked if m != "XGBoost"]
    order += [s for s in sorted(present) if s not in order]   # models absent from cv_metrics
    return [s for s in order if s in present]


def day_line_chart(plot_df, domain, height=380):
    """One delivery day, every series styled from SERIES_STYLE.

    Altair rather than st.line_chart because per-series width and dash are not
    expressible through st.line_chart. It is the same Vega-Lite renderer.
    """
    styles = [style(s) for s in domain]

    # draw order, bottom to top: the reference, then the baselines, then SARIMA
    # and XGBoost on top, so the model line is never buried under a baseline
    draw = [ACTUAL] + [s for s in reversed(domain) if s != ACTUAL]
    plot_df = plot_df.assign(_z=plot_df["series"].map({s: i for i, s in enumerate(draw)}))
    plot_df = plot_df.sort_values(["_z", "timestamp"])

    return (alt.Chart(plot_df)
              .mark_line()
              .encode(
                  x=alt.X("timestamp:T", title="Hour (Madrid time)"),
                  y=alt.Y("eur_mwh:Q", title=PRICE_AXIS),
                  color=alt.Color("series:N", title=None,
                                  scale=alt.Scale(domain=domain,
                                                  range=[s["color"] for s in styles]),
                                  legend=alt.Legend(orient="top", direction="horizontal")),
                  strokeWidth=alt.StrokeWidth("series:N", legend=None,
                                  scale=alt.Scale(domain=domain,
                                                  range=[s["width"] for s in styles])),
                  strokeDash=alt.StrokeDash("series:N", legend=None,
                                  scale=alt.Scale(domain=domain,
                                                  range=[s["dash"] for s in styles])),
              )
              .properties(height=height))



# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
@st.cache_data
def load_predictions():
    """Long CV table: timestamp, model, y_true, y_pred."""
    preds = pd.read_parquet(PREDICTIONS)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"])
    preds["day"] = preds["timestamp"].dt.date          # the delivery day
    preds["abs_error"] = (preds["y_true"] - preds["y_pred"]).abs()
    return preds


@st.cache_data
def load_metrics():
    """One row per (model, CV window): mae, rmse, mase."""
    return pd.read_csv(CV_METRICS)


@st.cache_data
def load_prices():
    """The hourly price history, on the same explicit grid the notebooks use."""
    prices = pd.read_csv(PRICES_CSV, parse_dates=["datetime"])
    prices = prices.set_index("datetime").sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    return prices.asfreq("h")


# st.cache_resource, not st.cache_data. st.cache_data is for values: it serialises
# what it caches and hands back a fresh copy on every call. A fitted booster is a
# live C++ object behind a Python handle, so it is the wrong shape for that.
# st.cache_resource returns the one same object to every rerun and every session,
# which is exactly what a loaded model wants.
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    return model


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("Spanish day-ahead electricity price forecasting")
st.caption("Backtested forecasts for the OMIE/ENTSO-E day-ahead auction. "
           "All results are frozen artifacts from the notebooks, nothing is trained here.")

if not PREDICTIONS.exists() or not CV_METRICS.exists():
    st.error("Missing artifacts. Run notebooks 03 and 04 to the end, they write "
             "data/processed/predictions.parquet and data/processed/cv_metrics.csv.")
    st.stop()

preds   = load_predictions()
metrics = load_metrics()
models  = sorted(preds["model"].unique())

with st.sidebar:
    st.header("Filters")
    selected = st.multiselect("Models", models, default=models)
    st.markdown("---")
    st.caption(f"Backtest: {preds['day'].min()} to {preds['day'].max()}  "
               f"({preds['day'].nunique()} delivery days)")
    st.caption("August only — one seasonal regime, results do not generalise to winter.")
    st.caption(f"[Source code on GitHub]({GITHUB_URL})")

preds = preds[preds["model"].isin(selected)]

tab_overview, tab_comparison, tab_forecast = st.tabs(
    ["Overview", "Model comparison", "Forecast"])


# --- Overview ---------------------------------------------------------------
with tab_overview:
    st.subheader("One delivery day at a time")

    days = sorted(preds["day"].unique())
    day  = st.selectbox("Delivery day", days, index=len(days) - 1)

    one_day = preds[preds["day"] == day]

    # long format: one row per (timestamp, series), so each line can carry its
    # own colour, width and dash from SERIES_STYLE
    modelled = (one_day[["timestamp", "model", "y_pred"]]
                .rename(columns={"model": "series", "y_pred": "eur_mwh"}))
    observed = (one_day.groupby("timestamp", as_index=False)["y_true"].first()
                       .rename(columns={"y_true": "eur_mwh"})
                       .assign(series=ACTUAL))
    plot_df = pd.concat([observed, modelled], ignore_index=True)

    domain = series_order(metrics, set(plot_df["series"]))
    st.altair_chart(day_line_chart(plot_df, domain), width="stretch")

    st.caption("The auction clears all 24 hours of D+1 at once, so every model "
               "predicts the whole profile in one shot.")

    cols = st.columns(max(len(selected), 1))
    for col, model in zip(cols, selected):
        col.metric(f"{model} MAE", f"{one_day.loc[one_day['model'] == model, 'abs_error'].mean():.2f}")


# --- Model comparison -------------------------------------------------------
with tab_comparison:
    st.subheader("Across all backtest days")

    left, right = st.columns(2)

    with left:
        st.markdown("**Mean absolute error by model** (EUR/MWh)")
        st.bar_chart(preds.groupby("model")["abs_error"].mean(), height=320,
                     x_label="Model", y_label=ERROR_AXIS)

    with right:
        st.markdown("**Mean absolute error by hour of day**")
        by_hour = (preds.assign(hour=preds["timestamp"].dt.hour)
                        .groupby(["hour", "model"])["abs_error"].mean().unstack("model"))
        st.line_chart(by_hour, height=320, x_label="Hour of day", y_label=ERROR_AXIS,
                      color=[style(c)["color"] for c in by_hour.columns])

    st.markdown("**Mean absolute error per delivery day**")
    per_day = preds.groupby(["day", "model"])["abs_error"].mean().unstack("model")
    st.line_chart(per_day, height=320, x_label="Delivery day", y_label=ERROR_AXIS,
                  color=[style(c)["color"] for c in per_day.columns])

    st.markdown("**Cross-validation metrics** (one row per model and window)")
    st.dataframe(metrics[metrics["model"].isin(selected)], width="stretch")

    st.markdown("**Averaged over all windows**")
    st.dataframe(metrics[metrics["model"].isin(selected)]
                 .groupby("model")[["mae", "rmse", "mase"]].mean().round(2),
                 width="stretch")


# --- Forecast ---------------------------------------------------------------
with tab_forecast:
    st.subheader("Next delivery day")

    if not MODEL_FILE.exists():
        st.info(f"No model at {MODEL_FILE}. Run notebook 04 to the end to write it.")
    else:
        prices   = load_prices()
        forecast = forecast_next_day(load_model(), prices)

        st.caption(f"Last known day: {prices.index.max().normalize().date()}  ->  "
                   f"forecast for {forecast.index[0].date()}")

        context = prices.loc[prices.index >= prices.index.max() - pd.Timedelta(days=7), [TARGET]]
        chart   = pd.concat([context.rename(columns={TARGET: "Observed"}),
                             forecast.rename(columns={"forecast_eur_mwh": "Forecast D+1"})])
        st.line_chart(chart, height=380, x_label="Hour (Madrid time)", y_label=PRICE_AXIS,
                      color=[style(ACTUAL)["color"], style("XGBoost")["color"]])

        st.dataframe(forecast.round(2), width="stretch")
