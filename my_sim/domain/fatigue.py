"""
Алгоритм усталости полностью отделён от SimPy — чистые функции и dataclass-ы.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import exp


# ─────── параметры, приходящие из YAML (dev_defaults) ────────────────────────
@dataclass(slots=True, frozen=True)
class FatigueParams:
    alpha: float          # скорость накопления tp при перегрузе
    beta_work: float      # скорость восстановления, если workload < w_thr
    beta_rest: float      # ↑ при праздниках/отпуске
    w_thr: float          # «комфортный» дневной объём, story-points
    tp0: float = 10.0     # начальный tire-points
    delta: float = 0.018  # крутизна влияния усталости на скорость


# ─────── модель ──────────────────────────────────────────────────────────────
class FatigueModel:
    """
    F_{t+1} = clamp(0,100, F_t  + α·max(0, w − w_thr)  − β·max(0, w_thr − w))
    """

    __slots__ = ("params", "tp")

    def __init__(self, params: FatigueParams):
        self.params = params
        self.tp: float = params.tp0

    # --- динамика ------------------------------------------------------------------
    def step(self, workload_sp: float, restoring: bool) -> None:
        """
        :param workload_sp: story-points, фактически сделанные за СУТКИ
                            (или любое ваше окно агрегации)
        :param restoring:   True = выходной/отпуск
        """
        p = self.params
        β = p.beta_rest if restoring else p.beta_work
        delta = p.alpha * max(0.0, workload_sp - p.w_thr) \
                - β * max(0.0, p.w_thr - workload_sp)
        self.tp = max(0.0, min(100.0, self.tp + delta))

    # --- влияние на скорость -------------------------------------------------------
    def speed_multiplier(self) -> float:
        """
        g(tp) = 1 / (1 + δ·tp)  ∈ (0,1]; никогда не даёт отрицательного результата.
        """
        return 1.0 / (1.0 + self.params.delta * self.tp)

    # --- удобные индикаторы --------------------------------------------------------
    def risk_burnout(self, kappa: float) -> float:
        """
        Вероятность выгорания (0…1) — сигмоида от средней tp за неделю.
        Kappa забирается из DevParams.
        """
        from math import exp
        return 1.0 / (1.0 + exp(-kappa * (self.tp / 100.0 - 0.6)))  # 0.6 ≈ «зона риска»

    def step_tick(self,
                  workload_sp_tick: float,
                  tick_fraction_of_day: float,
                  restoring: bool) -> None:
        """
        То же самое, но параметры автоматически масштабируются под длительность тика.
        """
        p = self.params
        w_thr_tick = p.w_thr * tick_fraction_of_day
        α = p.alpha * tick_fraction_of_day
        β_base = p.beta_rest if restoring else p.beta_work
        β = β_base * tick_fraction_of_day

        delta = α * max(0.0, workload_sp_tick - w_thr_tick) \
                - β * max(0.0, w_thr_tick - workload_sp_tick)
        self.tp = max(0.0, min(100.0, self.tp + delta))