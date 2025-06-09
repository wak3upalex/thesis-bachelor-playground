"""
SprintPlanner:
* распихивает Story-points между разработчиками,
* уважает общий WIP-лимит команды,
* вызывает .work_tick() у каждого dev раз в `tick_minutes`,
* складывает EventRow-словарь в done_q для Parquet-логера.

WIP-policy простая: “бери Story, куда влезет минимальный остаток”.
"""

from __future__ import annotations
import simpy
from collections import defaultdict, deque
from typing import List

from infrastructure.config import SimulationConfig
from infrastructure.logger import EventRow
from domain import Developer, Story


SPRINT_DAYS = 14   # две недели


def sprint_planner_process(env: simpy.Environment,
                           cfg: SimulationConfig,
                           backlog: simpy.Store,
                           done_q: simpy.Store,
                           devs: List[Developer]) -> simpy.events.Process:
    """
    :param backlog:  SimPy.Store[Story]
    :param done_q:   SimPy.Store[dict] для ParquetLogger
    :param devs:     список доменных Developer-объектов (НЕ SimPy процессов)
    """

    tick_sec = cfg.tick_minutes * 60
    tick_frac_day = cfg.tick_minutes / 1440.0

    # per-dev «остаток» работы (SP, float)
    remaining: dict[int, float] = {dev.pid: 0.0 for dev in devs}

    while True:
        now_dt = env.to_datetime(env.now)
        is_weekend = now_dt.weekday() >= 5
        sprint_idx = (now_dt.date() - cfg.start_date.date()).days // SPRINT_DAYS

        # ----- пополнить WIP --------------------------------------------------
        team_wip = sum(remaining.values())
        capacity = cfg.team.wip_limit - team_wip

        while capacity > 0 and backlog.items:
            story: Story = backlog.items.pop(0)       # FIFO
            sp = story.sp
            if sp > capacity:
                # не помещаемся → откатить назад и выходим
                backlog.items.insert(0, story)
                break

            # кому дать? выбираем dev с минимальным «хвостом»
            pid = min(remaining, key=remaining.get, default=0)
            remaining[pid] += sp
            capacity -= sp

        # ----- час работы каждого dev ----------------------------------------
        for dev in devs:
            pid = dev.pid
            workload = remaining[pid] if not is_weekend else 0.0

            done, tp = dev.work_tick(
                workload_sp=workload,
                tick_fraction_of_day=tick_frac_day,
                is_rest_day=is_weekend
            )

            remaining[pid] -= done

            done_q.put({
                "sim_time": env.now,
                "pid": pid,
                "sp_done": done,
                "tp_end": tp,
                "sprint": sprint_idx,
            })

        yield env.timeout(tick_sec)
