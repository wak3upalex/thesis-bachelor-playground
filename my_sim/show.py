import pandas as pd
import glob
from dateutil.parser import isoparse
from datetime import date, timedelta
import matplotlib.pyplot as plt

# 1. Загрузить конфиг и логи
from infrastructure.config import load_conf
from pathlib import Path
conf = load_conf(Path("config.yaml"))
start_date = conf.start_date

# все event-логи
paths = sorted(glob.glob("simlog/events-*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

# 2. Вычислить реальную дату события и ежедневную Velocity
df["date"] = start_date + pd.to_timedelta(df["sim_time"], unit="s")
daily = (df
    .groupby(df["date"].dt.normalize())["sp_done"]
    .sum()
    .reset_index()
    .rename(columns={"date": "day", "sp_done": "daily_sp"})
)

# 3. Помощник: конец квартала
def quarter_end(d: date) -> date:
    q_end_month = ((d.month - 1)//3)*3 + 3
    year = d.year + (q_end_month//12)
    month = q_end_month%12 + 1
    return date(year, month, 1) - timedelta(days=1)

# 4. Считаем дни до конца квартала
daily["days_to_q_end"] = daily["day"].dt.date.apply(
    lambda d: (quarter_end(d) - d).days
)

# (опционально) фильтр по одному кварталу: days_to_q_end <= 90
daily = daily[daily["days_to_q_end"] <= 90]

# 5. Рисуем scatter
plt.figure()
plt.scatter(daily["days_to_q_end"], daily["daily_sp"], marker="o")
plt.title("Daily Velocity vs. Days to Quarter End")
plt.xlabel("Days to Quarter End")
plt.ylabel("Story-Points Completed per Day")
plt.grid(alpha=0.3, linestyle="--", linewidth=0.4)
plt.tight_layout()
plt.show()
