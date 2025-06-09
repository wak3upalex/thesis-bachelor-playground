"""
Developer = актор, который:
1. Принимает workload (story-points) за тик.
2. Конвертирует их в «сделано» (может быть меньше, чем получил).
3. Обновляет усталость, знает переходы Burnout ↔ Recovery.

Этот класс НЕ знает о SimPy.  Любая среда (SimPy, asyncio, unit-тест) может
вызывать .work_tick(...) в своём цикле.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from random import Random, lognormvariate
from typing import Tuple

from .fatigue import FatigueModel, FatigueParams


class DeveloperState(Enum):
    WORKING = auto()
    BURNOUT = auto()


# ─────── параметры, приходящие из YAML (dev_defaults) ────────────────────────
@dataclass(slots=True, frozen=True)
class DevParams:
    s0: float               # базовая дневная скорость (SP/день)
    fatigue: FatigueParams  # вложенная структура
    lognorm_sigma: float    # дневной шум продуктивности
    kappa: float            # крутизна P(burnout)
    theta_recover: float    # tp-порог выхода из burnout


# ─────── актор ───────────────────────────────────────────────────────────────
class Developer:
    """
    Один вызов .work_tick() = 1 SimPy-тик (по умолчанию 1 час) ИЛИ любой
    ваш granular step — не важно, делите s0 из расчёта «сколько SP/тик».
    """

    __slots__ = (
        "pid", "_p", "_rng", "_state", "_fatigue", "_sp_done_total"
    )

    def __init__(self, pid: int, p: DevParams, seed: int | None = None) -> None:
        self.pid = pid
        self._p = p
        self._rng = Random(seed)
        self._state = DeveloperState.WORKING
        self._fatigue = FatigueModel(p.fatigue)
        self._sp_done_total: float = 0.0

    # -------------------------------------------------------------------------
    @property
    def state(self) -> DeveloperState:
        return self._state

    @property
    def tp(self) -> float:
        return self._fatigue.tp

    @property
    def sp_done_total(self) -> float:
        return self._sp_done_total

    # -------------------------------------------------------------------------
    def work_tick(self, workload_sp: float, tick_fraction_of_day: float,
                  is_rest_day: bool) -> Tuple[float, float]:
        """
        :param workload_sp:   сколько SP выдаёт планировщик на этот тик
        :param tick_fraction_of_day: 1/24 если тик = 1 час
        :param is_rest_day:   True = выходной → ↑восстановление
        :return: tuple( sp_done_this_tick, tp_after_tick )
        """
        # ----- burnout logic --------------------------------------------------
        if self._state is DeveloperState.BURNOUT:
            # попробовать выйти из burnout
            if self.tp < self._p.theta_recover * 100:
                self._state = DeveloperState.WORKING
            else:
                # отдыхает, ничего не делает
                self._fatigue.step(workload_sp=0.0, restoring=True)
                return 0.0, self.tp

        # ----- обычная работа -------------------------------------------------
        # baseline скорость → за тик
        speed_tick = self._p.s0 * tick_fraction_of_day
        speed_tick *= self._fatigue.speed_multiplier()

        # логнормальный шум через экземпляр Random
        mu = -0.5 * self._p.lognorm_sigma ** 2
        speed_tick *= self._rng.lognormvariate(mu, self._p.lognorm_sigma)

        done = min(workload_sp, speed_tick)
        self._sp_done_total += done

        # ----- обновить усталость --------------------------------------------
        self._fatigue.step_tick(workload_sp_tick=done,
                                 tick_fraction_of_day = tick_fraction_of_day,
        restoring = is_rest_day)

        # Проверить вероятность выгорания раз в конце суток (можно иначе)
        if is_rest_day is False:       # weekday
            p_burn = self._fatigue.risk_burnout(self._p.kappa)
            if self._rng.random() < p_burn:
                self._state = DeveloperState.BURNOUT

        return done, self.tp
