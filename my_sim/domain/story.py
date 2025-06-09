"""
Story = минимальная единица работы (обычно user-story или задача).
Здесь же — генератор story-points с хвостовым распределением (Zipf).
"""

from __future__ import annotations
from dataclasses import dataclass
from random import Random
from typing import List

import numpy as np


@dataclass(frozen=True, slots=True)
class Story:
    sp: int               # story-points
    created_at: float     # sim-time seconds (опционально)
    id: int | None = None


# ─────── распределение story-points ──────────────────────────────────────────
class StoryPointsDistribution:
    """
    P(k) ∝ k^-γ ,  k ∈ {1,2,…,k_max}.
    Реализовано через inverse-transform sampling (быстро при небольших k_max).
    """

    def __init__(self, gamma: float, k_max: int = 21, rng: Random | None = None):
        assert gamma > 0, "γ must be positive"
        self.gamma = gamma
        self.k_max = k_max
        self.rng = rng or Random()
        # норма-константа
        ks = np.arange(1, k_max + 1)
        self._cdf = np.cumsum(ks ** (-gamma)) / np.sum(ks ** (-gamma))

    def sample(self) -> int:
        u = self.rng.random()
        idx = np.searchsorted(self._cdf, u)
        return int(idx + 1)

    def sample_n(self, n: int) -> List[int]:
        return [self.sample() for _ in range(n)]


# ─────── удобная фабрика story-объектов ───────────────────────────────────────
class StoryFactory:
    """Создаёт Story с автогенерацией id и заданным распределением SP."""

    def __init__(self, sp_dist: StoryPointsDistribution):
        self.sp_dist = sp_dist
        self._counter = 0

    def make(self, now: float) -> Story:
        self._counter += 1
        return Story(
            sp=self.sp_dist.sample(),
            created_at=now,
            id=self._counter,
        )
