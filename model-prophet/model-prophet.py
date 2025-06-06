import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

try:
    from prophet import Prophet
except ImportError:
    print("Prophet не найден. Пожалуйста, установите пакет: \n  pip install prophet")
    raise


def load_and_prepare(tasks_path: str,
                     events_path: str,
                     employee: str | None = None,
                     week_start: str = "Mon"):
    """
    Загружает CSV, агрегирует данные на уровне недели, возвращает dataframe, готовый для Prophet.
    """
    tasks = pd.read_csv(tasks_path, parse_dates=["date"])
    if employee:
        tasks = tasks[tasks["employee"] == employee]
    if "tasks_done" not in tasks.columns:
        tasks["tasks_done"] = 1
    daily_tasks = tasks.groupby("date", as_index=False)["tasks_done"].sum()

    offset = {"Mon": 0, "Sun": 6}[week_start]
    daily_tasks["week_start"] = daily_tasks["date"] - pd.to_timedelta(
        (daily_tasks["date"].dt.weekday - offset) % 7, unit="D"
    )
    weekly = daily_tasks.groupby("week_start", as_index=False)["tasks_done"].sum()
    df = weekly.rename(columns={"week_start": "ds", "tasks_done": "y"})

    events = pd.read_csv(events_path, parse_dates=["date"])
    if events.empty:
        regressors = pd.DataFrame({"ds": df["ds"]})
    else:
        events["week_start"] = events["date"] - pd.to_timedelta(
            (events["date"].dt.weekday - offset) % 7, unit="D"
        )
        events["flag"] = 1
        weekly_events = events.groupby(["week_start", "event_type"])["flag"].max().unstack(
            fill_value=0).reset_index()
        weekly_events = weekly_events.rename(columns={"week_start": "ds"})
        regressors = weekly_events

    df = df.merge(regressors, on="ds", how="left").fillna(0)
    return df, regressors.columns.drop("ds").tolist()


def train_prophet(df: pd.DataFrame, regressors: list[str]):
    """
    Тренировка модели Prophet с регрессорами.
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
    Создание будущего dataframe для предсказания, включая регрессоры для будущих событий.
    """
    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    future = future.merge(regressors_df, on="ds", how="left").fillna(0)
    for r in regressors_cols:
        if r not in future.columns:
            future[r] = 0
    return future


def forecast(tasks_csv: str,
             events_csv: str,
             forecast_weeks: int = 8,
             employee: str | None = None):
    """
    Полный модуль предсказания:
    загрузка данных, тренировка Prophet, предсказание и выдача графика результатов.
    Возвращает предсказанные данные в виде dataframe.
    """
    df, reg_cols = load_and_prepare(tasks_csv, events_csv, employee)
    m = train_prophet(df, reg_cols)

    events = pd.read_csv(events_csv, parse_dates=["date"])
    offset = 0
    events["week_start"] = events["date"] - pd.to_timedelta(
        (events["date"].dt.weekday - offset) % 7, unit="D"
    )
    events["flag"] = 1
    future_events = events.groupby(["week_start", "event_type"])["flag"].max().unstack(
        fill_value=0).reset_index()
    future_events = future_events.rename(columns={"week_start": "ds"})

    future = make_future_dataframe(m, forecast_weeks, future_events, reg_cols)
    fcst = m.predict(future)

    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    m.plot(fcst, ax=ax)
    ax.set_title("Еженедельный прогноз по выполнению задач")
    plt.show()

    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]


tasks_path = "tasks.csv"
events_path = "events.csv"
result = forecast(tasks_path, events_path, forecast_weeks=4, employee=None)
display(result)
