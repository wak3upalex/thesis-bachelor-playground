import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 100 days of tasks data from 2025-01-01
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(100)]

# Simulate tasks_done: base around 5 with some noise, dips on event days we'll mark later
np.random.seed(42)
tasks_done = np.random.poisson(lam=5, size=100)

tasks_df = pd.DataFrame({
    'date': dates,
    'tasks_done': tasks_done
})

# Generate events: holidays every 25 days and exam every 40 days
events = []
event_types = ['Holiday', 'Exam']
for i, d in enumerate(dates):
    if i % 25 == 0:
        events.append({'date': d, 'event_type': 'Holiday'})
        tasks_df.loc[tasks_df['date'] == d, 'tasks_done'] = max(0, tasks_df.loc[tasks_df['date'] == d, 'tasks_done'].iloc[0] - 3)
    if i % 40 == 0:
        events.append({'date': d, 'event_type': 'Exam'})
        tasks_df.loc[tasks_df['date'] == d, 'tasks_done'] = max(0, tasks_df.loc[tasks_df['date'] == d, 'tasks_done'].iloc[0] - 2)

events_df = pd.DataFrame(events)

# Save to CSV
tasks_df.to_csv('tasks.csv', index=False)
events_df.to_csv('events.csv', index=False)
