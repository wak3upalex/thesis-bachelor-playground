"""
Daily Non-Homogeneous Poisson arrival of Stories.

* λ_base   — зависит от λ0 и размера команды,
* holiday  — в выходные/праздники поток ×(1-drop),
* urgency  — экспонента к концу квартала: 1 + A·exp(-(days_to_Qend)/τ),
* Story-объекты кладутся в SimPy.Store backlog.
"""

from __future__ import annotations
import simpy
import numpy as np
from datetime import timedelta, date, datetime

from domain import StoryFactory, StoryPointsDistribution
from infrastructure.config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
def _quarter_end(d: date) -> date:
    """Вернёт последний день текущего квартала."""
    q_end_month = ((d.month - 1) // 3) * 3 + 3
    # 1-е число следующего месяца, потом минус 1 день
    next_month = datetime(d.year + (q_end_month // 12),
                          q_end_month % 12 + 1, 1).date()
    return next_month - timedelta(days=1)


def _days_to_quarter_end(d: date) -> int:
    return (_quarter_end(d) - d).days


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def task_arrival_process(env: simpy.Environment,
                         cfg: SimulationConfig,
                         backlog: simpy.Store) -> simpy.events.Process:
    """
    Один шаг = один календарный день.
    На каждый SP создаётся Story-объект (Zipf-размер), который кладётся
    в backlog.
    """

    sp_dist = StoryPointsDistribution(gamma=cfg.story_points.zipf_gamma)
    factory = StoryFactory(sp_dist)

    λ0 = cfg.arrival.lambda0 * cfg.team.n_devs
    A = cfg.arrival.amplitude_quarter_end
    τ = cfg.arrival.tau
    drop = cfg.arrival.holiday_drop

    day_sec = 24 * 3600

    while True:
        today = env.to_datetime(env.now).date()

        # ----- интенсивность потока -----------------------------------------
        base = λ0
        urgency = 1.0 + A * np.exp(-_days_to_quarter_end(today) / τ)
        holiday_mult = 1.0 - (drop if _is_weekend(today) else 0.0)

        lam = base * urgency * holiday_mult
        total_sp = np.random.poisson(lam)

        # ----- сгенерировать истории ----------------------------------------
        for _ in range(total_sp):
            story = factory.make(env.now)
            backlog.put(story)

        yield env.timeout(day_sec)
