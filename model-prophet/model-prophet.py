import pandas as pd
import numpy as np
from datetime import timedelta
from IPython.display import display
import matplotlib.pyplot as plt
from prophet import Prophet

def load_and_prepare(tasks_path: str,
                     events_path: str,
                     employee: str | None = None,
                     week_start: str = "Mon"):
    """
    Load CSVs, aggregate to weekly level, return DataFrame for Prophet.
    """
    # Load tasks
    tasks = pd.read_csv(tasks_path, parse_dates=["date"])
    if employee:
        tasks = tasks[tasks["employee"] == employee]
    if "tasks_done" not in tasks.columns:
        tasks["tasks_done"] = 1
    daily = tasks.groupby("date", as_index=False)["tasks_done"].sum()

    # Align to week start
    offset = {"Mon": 0, "Sun": 6}[week_start]
    daily["week_start"] = daily["date"] - pd.to_timedelta(
        (daily["date"].dt.weekday - offset) % 7, unit="D"
    )
    weekly = daily.groupby("week_start", as_index=False)["tasks_done"].sum()
    df = weekly.rename(columns={"week_start": "ds", "tasks_done": "y"})

    # Load events
    events = pd.read_csv(events_path, parse_dates=["date"])
    if not events.empty:
        events["week_start"] = events["date"] - pd.to_timedelta(
            (events["date"].dt.weekday - offset) % 7, unit="D"
        )
        events["flag"] = 1
        weekly_events = (
            events.groupby(["week_start", "event_type"])["flag"]
            .max()
            .unstack(fill_value=0)
            .reset_index()
        ).rename(columns={"week_start": "ds"})
        df = df.merge(weekly_events, on="ds", how="left").fillna(0)
        reg_cols = weekly_events.columns.drop("ds").tolist()
    else:
        reg_cols = []
    return df, reg_cols

def train_prophet(df: pd.DataFrame, regressors: list[str]):
    """
    Train Prophet with Russian holidays and extra seasonalities.
    """
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5
    )
    m.add_country_holidays(country_name='RU')
    # custom monthly seasonality for extra variability
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    for r in regressors:
        m.add_regressor(r)
    m.fit(df)
    return m

def make_future_dataframe(model: Prophet,
                          periods: int,
                          regressors_df: pd.DataFrame,
                          regressors_cols: list[str]):
    """
    Create future DataFrame including regressors for future events.
    """
    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    future = future.merge(regressors_df, on="ds", how="left").fillna(0)
    for r in regressors_cols:
        if r not in future:
            future[r] = 0
    return future

def forecast(tasks_csv: str,
             events_csv: str,
             forecast_weeks: int = 12,
             employee: str | None = None):
    """
    Full pipeline: load data, train, forecast, visualize.
    """
    df, reg_cols = load_and_prepare(tasks_csv, events_csv, employee)
    m = train_prophet(df, reg_cols)

    # prepare future events
    events = pd.read_csv(events_csv, parse_dates=["date"])
    offset = 0
    events["week_start"] = events["date"] - pd.to_timedelta(
        (events["date"].dt.weekday - offset) % 7, unit="D"
    )
    events["flag"] = 1
    future_events = (
        events.groupby(["week_start", "event_type"])["flag"]
        .max()
        .unstack(fill_value=0)
        .reset_index()
    ).rename(columns={"week_start": "ds"})

    future = make_future_dataframe(m, forecast_weeks, future_events, reg_cols)
    fcst = m.predict(future)

    fig, ax = plt.subplots(figsize=(10, 6))
    m.plot(fcst, ax=ax)
    ax.set_title("Weekly Task Forecast (with Russian Holidays)")
    plt.show()

    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# Example usage:
result = forecast("tasks.csv", "events.csv", forecast_weeks=52, employee=None)
display(result.head())
