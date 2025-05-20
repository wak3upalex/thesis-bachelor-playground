import pandas as pd
import numpy as np
from IPython.display import display

# Minimal working Prophet-based forecaster for weekly task completion
# -------------------------------------------------------------------
# Expected CSV formats:
# 1. tasks.csv       -> columns: date, tasks_done[, employee]
#                     each row = one task done date OR aggregated count per day
# 2. events.csv      -> columns: date, event_type
#                     each row = date of an event (holiday, exam, etc.)
#
# The script:
# • aggregates tasks by week
# • aggregates events by week and creates binary regressors per event_type
# • trains Prophet model with extra regressors
# • forecasts future weekly task counts taking into account future events
# • plots the forecast
#
# Usage: modify the file paths and parameters at the end of this cell;
# run the cell – a plot and the forecast table will appear.
#
# NOTE: Install Prophet in your environment before running:
# pip install prophet
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

try:
    from prophet import Prophet
except ImportError:
    print("Prophet not found. Please install it with:\n  pip install prophet")
    raise

def load_and_prepare(tasks_path: str,
                     events_path: str,
                     employee: str | None = None,
                     week_start: str = "Mon"):
    """
    Load CSVs, aggregate to weekly level, return dataframe ready for Prophet.
    """
    # Load tasks
    tasks = pd.read_csv(tasks_path, parse_dates=["date"])
    if employee:
        tasks = tasks[tasks["employee"] == employee]
    # if tasks per row isn't count, aggregate by date
    if "tasks_done" not in tasks.columns:
        tasks["tasks_done"] = 1
    daily_tasks = tasks.groupby("date", as_index=False)["tasks_done"].sum()

    # Convert to weekly (Prophet expects 'ds' and 'y')
    # Align to week_start (Mon or Sun)
    offset = {"Mon": 0, "Sun": 6}[week_start]
    daily_tasks["week_start"] = daily_tasks["date"] - pd.to_timedelta(
        (daily_tasks["date"].dt.weekday - offset) % 7, unit="D"
    )
    weekly = daily_tasks.groupby("week_start", as_index=False)["tasks_done"].sum()
    df = weekly.rename(columns={"week_start": "ds", "tasks_done": "y"})

    # Load events, aggregate to same weekly index, pivot to dummy columns
    events = pd.read_csv(events_path, parse_dates=["date"])
    if events.empty:
        regressors = pd.DataFrame({"ds": df["ds"]})
    else:
        events["week_start"] = events["date"] - pd.to_timedelta(
            (events["date"].dt.weekday - offset) % 7, unit="D"
        )
        events["flag"] = 1
        weekly_events = events.groupby(["week_start", "event_type"])["flag"].max().unstack(fill_value=0).reset_index()
        weekly_events = weekly_events.rename(columns={"week_start": "ds"})
        regressors = weekly_events

    df = df.merge(regressors, on="ds", how="left").fillna(0)
    return df, regressors.columns.drop("ds").tolist()

def train_prophet(df: pd.DataFrame, regressors: list[str]):
    """
    Train Prophet with given regressors.
    """
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    for r in regressors:
        m.add_regressor(r)
    m.fit(df)
    return m

def make_future_dataframe(model: Prophet,
                          periods: int,
                          regressors_df: pd.DataFrame,
                          regressors_cols: list[str]):
    """
    Create future dataframe for forecast, including regressors for future events.
    """
    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    future = future.merge(regressors_df, on="ds", how="left").fillna(0)
    # Ensure all regressor cols exist
    for r in regressors_cols:
        if r not in future.columns:
            future[r] = 0
    return future

def forecast(tasks_csv: str,
             events_csv: str,
             forecast_weeks: int = 8,
             employee: str | None = None):
    """
    Full pipeline: load data, train Prophet, forecast, plot result.
    Returns forecast dataframe.
    """
    df, reg_cols = load_and_prepare(tasks_csv, events_csv, employee)
    m = train_prophet(df, reg_cols)

    # Prepare regressors for future (events_df may have future events)
    events = pd.read_csv(events_csv, parse_dates=["date"])
    offset = 0  # week start Monday
    events["week_start"] = events["date"] - pd.to_timedelta(
        (events["date"].dt.weekday - offset) % 7, unit="D"
    )
    events["flag"] = 1
    future_events = events.groupby(["week_start", "event_type"])["flag"].max().unstack(fill_value=0).reset_index()
    future_events = future_events.rename(columns={"week_start": "ds"})

    future = make_future_dataframe(m, forecast_weeks, future_events, reg_cols)
    fcst = m.predict(future)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    m.plot(fcst, ax=ax)
    ax.set_title("Weekly tasks forecast")
    plt.show()

    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# ----------------- EXAMPLE RUN -----------------
# Provide your own CSV paths here. Synthetic examples to illustrate:
#
# Suppose tasks.csv:
# date,tasks_done,employee
# 2024-01-01,3,alice
# 2024-01-02,2,alice
# ...
#
# and events.csv:
# date,event_type
# 2024-01-01,NewYear
# 2024-02-23,DefendersDay
#
# events.csv may include future dates (e.g., upcoming holidays).
#
# Uncomment and set paths to run:

tasks_path = "tasks.csv"
events_path = "events.csv"
result = forecast(tasks_path, events_path, forecast_weeks=50, employee=None)
display(result)
