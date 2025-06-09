"""
Thin wrapper around SimPy Environment:
* хранит tick-гранулярность и дату старта,
* даёт helper   env.to_datetime(sim_time_sec) → datetime,
* больше ничего не знает.
"""

from __future__ import annotations
import simpy
from datetime import timedelta, datetime
from infrastructure.config import SimulationConfig


def make_environment(cfg: SimulationConfig) -> simpy.Environment:
    """
    Создаёт SimPy Environment и «впрыскивает» в него:
      - cfg.start_date: календарная дата начала
      - env.tick_seconds: сколько секунд в одном тике
      - env.to_datetime(sec) → datetime
    """
    env = simpy.Environment(initial_time=0.0)
    env.tick_seconds = cfg.tick_minutes * 60
    env.start_date: datetime = cfg.start_date

    # Перевод из сим-секунд в реальную дату
    env.to_datetime = lambda secs: cfg.start_date + timedelta(seconds=secs)  # type: ignore

    return env
